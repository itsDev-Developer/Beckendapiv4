"""Peer resolution helpers -- fixes 'Peer id invalid' errors.

Drop this in as bot/helper/peers.py

Two separate problems, both real:

1. Pyrogram can only resolve a numeric chat_id from its own local peer
   cache (access hash storage). That cache is populated by receiving a live
   update mentioning the peer, or by resolving a public @username, or --
   for USER accounts only -- by calling get_dialogs(). Bots CANNOT call
   get_dialogs() at all; Telegram rejects it with BOT_METHOD_INVALID no
   matter how the bot is configured. So a freshly started bot client (a
   MULTI_TOKEN worker, or StreamBot after a restart with no session
   persistence) has no general-purpose way to force-resolve a private
   channel it hasn't seen a live update from yet in the current session.

2. When resolution fails locally (peer simply not in the cache), Pyrogram
   raises a plain ValueError -- NOT pyrogram.errors.PeerIdInvalid, which is
   the *server-side* RPC error for a different failure mode (a technically
   cached but mismatched peer). Code that only catches PeerIdInvalid will
   silently never retry.

What this module does about each:

- For USER accounts (UserBot), get_dialogs() works and is used to warm up
  every configured channel at once.
- For BOT accounts, get_dialogs() is skipped (logged at INFO, not treated
  as an error) and, if you provide an invite link for a channel via
  AUTH_CHANNEL_INVITE_LINKS, the bot joins/re-joins via that link instead --
  which *is* bot-legal and resolves+caches the peer without needing any
  prior state. Without an invite link, a bot channel can only become
  resolvable by actually receiving a live update (e.g. a new message posted
  while it's running, or being (re-)added to the channel while online) --
  there is no further trick around this; it's a hard protocol limitation.
- Both ValueError("Peer id invalid: ...") and pyrogram.errors.PeerIdInvalid
  are treated as "unresolved peer" everywhere in this module.
"""

import asyncio
import logging

from pyrogram.errors import ChannelPrivate, PeerIdInvalid

LOGGER = logging.getLogger(__name__)


def _is_unresolved_peer_error(e: Exception) -> bool:
    """True for both the server-side RPC error and the client-side
    ValueError Pyrogram raises when a chat_id isn't in its local cache."""
    if isinstance(e, PeerIdInvalid):
        return True
    return isinstance(e, ValueError) and "peer id invalid" in str(e).lower()


def _is_bot_method_invalid(e: Exception) -> bool:
    return "BOT_METHOD_INVALID" in str(e)


async def warm_up_peers(client, channel_ids, label="client", invite_links=None):
    """Resolve every configured channel for this client, as best as possible.

    Call this once, right after client.start(), for StreamBot, UserBot,
    and every multi-client -- before it's put to any real use.

    invite_links: optional {str(channel_id): invite_link} mapping (see
    Telegram.AUTH_CHANNEL_INVITE_LINKS). Only useful for bot clients, since
    it's the one bot-legal way to force a fresh resolve with no prior state.
    """
    invite_links = invite_links or {}
    seen = set()

    try:
        async for dialog in client.get_dialogs():
            seen.add(dialog.chat.id)
    except Exception as e:
        if _is_bot_method_invalid(e):
            LOGGER.info(
                "%s: get_dialogs() isn't available to bot accounts (this is "
                "expected) -- falling back to invite links / live updates.",
                label,
            )
        else:
            LOGGER.error("%s: get_dialogs failed while warming up peers: %s", label, e)

    for channel_id in channel_ids:
        if int(channel_id) in seen:
            continue

        link = invite_links.get(str(channel_id))
        if link:
            try:
                await client.join_chat(link)
                LOGGER.info("%s: resolved channel %s via invite link", label, channel_id)
                continue
            except Exception as e:
                if "PARTICIPANT" in str(e).upper():
                    # Already a member -- Telegram's response for this still
                    # carries the chat's full info, which Pyrogram caches on
                    # the way to raising this, so treat it as a resolve.
                    LOGGER.info("%s: already a member of %s (resolved via invite link)", label, channel_id)
                    continue
                LOGGER.warning(
                    "%s: could not resolve %s via its invite link (%s); "
                    "falling back to waiting for a live update.",
                    label, channel_id, e,
                )

        LOGGER.warning(
            "%s: could not eagerly resolve channel %s (not in dialogs, no "
            "working invite link configured). If this is a bot client, it "
            "will only become resolvable once it receives a live update "
            "from that channel (e.g. a new message) while running, or you "
            "add an entry for it to AUTH_CHANNEL_INVITE_LINKS.",
            label, channel_id,
        )


async def warm_up_all(clients_with_labels, channel_ids, invite_links=None):
    """Warm up several clients concurrently at startup."""
    await asyncio.gather(*[
        warm_up_peers(client, channel_ids, label=label, invite_links=invite_links)
        for client, label in clients_with_labels
    ])


async def _rescan_dialogs(client):
    """Best-effort re-scan; silently does nothing for bot clients (expected)."""
    try:
        async for _ in client.get_dialogs():
            pass
    except Exception as e:
        LOGGER.debug("get_dialogs unavailable for this client (expected for bots): %s", e)


async def safe_get_messages(client, chat_id, message_id, _retried=False):
    """get_messages with one automatic re-resolve-and-retry on an unresolved peer.

    Use this in place of client.get_messages(...) anywhere a chat_id might
    not have been resolved yet -- e.g. right after an admin changes
    AUTH_CHANNEL via /config without restarting the service. Note this can
    only help if something (a rescan, a prior update) actually resolves the
    peer between the two attempts -- for a bot with no invite link and no
    live update yet, the retry will fail identically and re-raise.
    """
    try:
        return await client.get_messages(chat_id, message_id)
    except Exception as e:
        if not _is_unresolved_peer_error(e):
            raise
        if _retried:
            raise
        LOGGER.info("Peer %s unresolved, re-scanning dialogs and retrying", chat_id)
        await _rescan_dialogs(client)
        return await safe_get_messages(client, chat_id, message_id, _retried=True)


async def safe_get_chat(client, chat_id, _retried=False):
    try:
        return await client.get_chat(chat_id)
    except ChannelPrivate as e:
        LOGGER.error("Client has no access to %s: %s", chat_id, e)
        raise
    except Exception as e:
        if not _is_unresolved_peer_error(e):
            raise
        if _retried:
            raise
        LOGGER.info("Peer %s unresolved, re-scanning dialogs and retrying", chat_id)
        await _rescan_dialogs(client)
        return await safe_get_chat(client, chat_id, _retried=True)


async def safe_iter(client, agen_factory, _retried=False):
    """Wrap an async-generator Telegram call (search_messages, get_chat_history)
    with the same re-resolve-and-retry behaviour.

    agen_factory is a zero-arg callable that returns a *fresh* async
    generator each time it's called, e.g.:

        safe_iter(UserBot, lambda: UserBot.get_chat_history(chat_id=..., limit=50))
    """
    try:
        async for item in agen_factory():
            yield item
    except Exception as e:
        if not _is_unresolved_peer_error(e):
            raise
        if _retried:
            raise
        LOGGER.info("Peer unresolved mid-iteration, re-scanning dialogs and retrying")
        await _rescan_dialogs(client)
        async for item in safe_iter(client, agen_factory, _retried=True):
            yield item
