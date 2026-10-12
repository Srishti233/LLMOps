"""Programmatic instruction-dataset generator: customer support for the fictional product AuroraSync.

No external download is needed. Every example carries `keywords`: the key facts the answer must
contain, which the evaluation step uses for its keyword-match metric.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

PRODUCT = "AuroraSync"

PLANS: dict[str, dict[str, Any]] = {
    "Free": {"storage": "5 GB", "price": "$0 and never expires", "upload": "2 GB", "trash_days": 30, "devices": 2},
    "Plus": {"storage": "200 GB", "price": "$4.99 per month", "upload": "10 GB", "trash_days": 90, "devices": 5},
    "Business": {"storage": "2 TB per user", "price": "$12 per user per month", "upload": "50 GB",
                 "trash_days": 180, "devices": 20},
}
PAID_PLANS = ["Plus", "Business"]

PLATFORMS: dict[str, dict[str, str]] = {
    "Windows": {"store": "aurorasync.example/download", "install": "the .exe installer from aurorasync.example/download",
                "tray": "system tray"},
    "macOS": {"store": "aurorasync.example/download", "install": "the .dmg from aurorasync.example/download",
              "tray": "menu bar"},
    "Linux": {"store": "aurorasync.example/download",
              "install": "the .deb package or AppImage from aurorasync.example/download", "tray": "system tray"},
    "Android": {"store": "Google Play", "install": "the app from Google Play", "tray": "notification shade"},
    "iOS": {"store": "App Store", "install": "the app from the App Store", "tray": "status bar"},
}
MOBILE = ["Android", "iOS"]

# slots: which parameter sets a topic needs ("plan", "paid_plan", "platform", "mobile")
TOPICS: list[dict[str, Any]] = [
    {"name": "reset_password", "slots": [],
     "questions": ["How do I reset my {product} password?",
                   "I forgot my password. How can I get back into my account?",
                   "What is the process to change a lost password?",
                   "I can't log in because I forgot my password, what should I do?"],
     "answer": "Go to Settings > Account > Reset Password, or choose 'Forgot password' on the sign-in screen. "
               "Enter your account email and we will send a reset link. The link expires in 30 minutes, so use it promptly.",
     "keywords": ["Reset Password", "30 minutes"]},
    {"name": "storage_limit", "slots": ["plan"],
     "questions": ["How much storage do I get on the {plan} plan?", "What is the storage limit for {plan}?",
                   "I'm on {plan}. How much space do I have?"],
     "answer": "The {plan} plan includes {storage} of storage. You can see your current usage under "
               "Settings > Account > Storage.",
     "keywords": ["{storage}", "Settings > Account > Storage"]},
    {"name": "pricing", "slots": ["plan"],
     "questions": ["How much does the {plan} plan cost?", "What is the price of {plan}?",
                   "Can you tell me what {plan} costs?"],
     "answer": "The {plan} plan costs {price}. You can compare plans or switch at Settings > Billing > Change Plan.",
     "keywords": ["{price}", "Change Plan"]},
    {"name": "refund", "slots": [],
     "questions": ["Can I get a refund?", "I bought a plan by mistake. How do I get my money back?",
                   "What is your refund policy?"],
     "answer": "You can request a refund within 30 days of purchase at Settings > Billing > Request Refund. "
               "Refunds go back to the original payment method within 5 business days.",
     "keywords": ["30 days", "Request Refund"]},
    {"name": "cancel", "slots": ["paid_plan"],
     "questions": ["How do I cancel my {plan} subscription?", "I want to stop paying for {plan}. What do I do?",
                   "Where can I cancel the {plan} plan?"],
     "answer": "Go to Settings > Billing > Cancel Plan. You keep your {plan} features until the end of the current "
               "billing period, and your files stay safe.",
     "keywords": ["Cancel Plan", "end of the current billing period"]},
    {"name": "sync_stuck", "slots": ["platform"],
     "questions": ["My files are not syncing on {platform}. What can I do?",
                   "Sync is stuck on my {platform} device. How do I fix it?",
                   "{product} on {platform} keeps saying 'syncing' forever."],
     "answer": "On {platform}, open the {tray} icon and choose Pause, then Resume Sync. If it is still stuck, go to "
               "Settings > Sync > Rescan Folders and make sure you have a stable connection and free storage.",
     "keywords": ["Resume Sync", "Rescan Folders"]},
    {"name": "install", "slots": ["platform"],
     "questions": ["How do I install {product} on {platform}?", "Where can I download {product} for {platform}?",
                   "I need the {product} app for {platform}. How do I get it?"],
     "answer": "On {platform}, get {install}. Then sign in with your {product} account and pick a folder to sync.",
     "keywords": ["{store}", "sign in"]},
    {"name": "share_link", "slots": [],
     "questions": ["How do I share a file with someone?", "Can I send a file to a friend using a link?",
                   "What is the way to share a folder with my colleague?"],
     "answer": "Right-click the file and choose Share > Create Link. You can set an expiry date and a password "
               "for the link. Anyone with the link can view the file.",
     "keywords": ["Share > Create Link", "expiry date"]},
    {"name": "two_factor", "slots": [],
     "questions": ["How do I turn on two-factor authentication?", "Can I make my account more secure with 2FA?",
                   "Where do I enable two step verification?"],
     "answer": "Go to Settings > Security > Two-Factor Authentication and scan the QR code with an authenticator app. "
               "Save your 10 backup codes in a safe place.",
     "keywords": ["Two-Factor Authentication", "backup codes"]},
    {"name": "recover_deleted", "slots": ["plan"],
     "questions": ["I deleted a file by accident on {plan}. Can I get it back?",
                   "How do I restore a deleted file on the {plan} plan?",
                   "Is there a recycle bin on {plan}?"],
     "answer": "Open Trash in the sidebar and click Restore. On the {plan} plan, deleted files stay in Trash for "
               "{trash_days} days.",
     "keywords": ["Trash", "{trash_days} days"]},
    {"name": "contact_support", "slots": [],
     "questions": ["How do I contact support?", "What is the best way to reach a human agent?",
                   "When is your support team available?"],
     "answer": "Email help@aurorasync.example or use Help > Contact Support in the app. Our team is available "
               "Mon-Fri 9am-6pm UTC and replies within one business day.",
     "keywords": ["help@aurorasync.example", "Mon-Fri 9am-6pm UTC"]},
    {"name": "upload_limit", "slots": ["plan"],
     "questions": ["What is the maximum file size I can upload on {plan}?",
                   "My big video will not upload on the {plan} plan. What is the size limit?",
                   "Is there a file size limit for {plan}?"],
     "answer": "On the {plan} plan, the maximum file size per upload is {upload}.",
     "keywords": ["{upload}"]},
    {"name": "offline_access", "slots": ["mobile"],
     "questions": ["How do I use my files offline on {platform}?",
                   "Can I open files without internet on {platform}?",
                   "How can I keep a file available offline on my {platform} phone?"],
     "answer": "On {platform}, tap the three-dot menu next to a file and choose Make Available Offline. "
               "Offline files sync again when you reconnect.",
     "keywords": ["Make Available Offline"]},
    {"name": "delete_account", "slots": [],
     "questions": ["How do I delete my account?", "I want to close my {product} account permanently. What are the steps?"],
     "answer": "Go to Settings > Account > Delete Account and confirm with your password. Your data is permanently "
               "deleted after 14 days; contact support within that time if you change your mind.",
     "keywords": ["Delete Account", "14 days"]},
    {"name": "add_members", "slots": [],
     "questions": ["How do I add team members to my Business account?", "Can I invite my coworkers to {product} Business?",
                   "Where do I manage users on the Business plan?"],
     "answer": "Open Admin Console > Members > Invite and enter their email addresses. Business accounts support "
               "up to 500 members.",
     "keywords": ["Admin Console > Members > Invite", "500 members"]},
    {"name": "devices", "slots": ["plan"],
     "questions": ["How many devices can I link on {plan}?", "What is the device limit for the {plan} plan?",
                   "I'm on {plan}. Can I use {product} on several computers?"],
     "answer": "The {plan} plan lets you link up to {devices} devices. Manage them at Settings > Security > Devices.",
     "keywords": ["{devices} devices", "Settings > Security > Devices"]},
    {"name": "version_history", "slots": ["plan"],
     "questions": ["Does {product} keep older versions of my files on {plan}?",
                   "How do I go back to a previous version of a document on the {plan} plan?"],
     "answer": "Yes. Right-click a file and choose Version History to restore an earlier version. On the {plan} plan "
               "versions are kept for {trash_days} days.",
     "keywords": ["Version History", "{trash_days} days"]},
    {"name": "selective_sync", "slots": [],
     "questions": ["How do I stop a huge folder from syncing to my laptop?",
                   "I want some folders to stay in the cloud only. Is that possible?"],
     "answer": "Use Settings > Sync > Selective Sync and untick the folder. It stays in the cloud but is removed "
               "from this device.",
     "keywords": ["Selective Sync"]},
    {"name": "invoice", "slots": [],
     "questions": ["Where can I download my invoice?", "I need a receipt for my subscription. Where do I find it?"],
     "answer": "Go to Settings > Billing > Invoices and click Download PDF next to the invoice you need.",
     "keywords": ["Billing > Invoices", "Download PDF"]},
]

PREFIXES = ["", "", "", "Hi, ", "Hello, ", "Hi team, ", "Quick question: ", "Hey! "]
SUFFIXES = ["", "", "", " Thanks!", " Please help.", " Thank you."]


def _slot_values(slots: list[str], rng: random.Random) -> dict[str, Any]:
    values: dict[str, Any] = {"product": PRODUCT}
    for slot in slots:
        if slot == "plan":
            name = rng.choice(list(PLANS))
            values.update(PLANS[name], plan=name)
        elif slot == "paid_plan":
            name = rng.choice(PAID_PLANS)
            values.update(PLANS[name], plan=name)
        elif slot == "platform":
            name = rng.choice(list(PLATFORMS))
            values.update(PLATFORMS[name], platform=name)
        elif slot == "mobile":
            name = rng.choice(MOBILE)
            values.update(PLATFORMS[name], platform=name)
    return values


def generate_examples(n: int = 300, seed: int = 42) -> list[dict[str, Any]]:
    """Generate `n` unique examples, cycling through topics so all are covered."""
    rng = random.Random(seed)
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    attempts = 0
    while len(examples) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Could not generate enough unique examples; reduce n.")
        topic = TOPICS[len(examples) % len(TOPICS)]
        values = _slot_values(topic["slots"], rng)
        question = rng.choice(topic["questions"]).format(**values)
        instruction = f"{rng.choice(PREFIXES)}{question}{rng.choice(SUFFIXES)}".strip()
        if instruction in seen:
            continue
        seen.add(instruction)
        examples.append({
            "id": f"ex{len(examples):04d}",
            "topic": topic["name"],
            "instruction": instruction,
            "response": topic["answer"].format(**values),
            "keywords": [kw.format(**values) for kw in topic["keywords"]],
            "source": "synthetic",
        })
    return examples


def split_examples(examples: list[dict[str, Any]], val_fraction: float = 0.15,
                   seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_fraction)))
    return rows[n_val:], rows[:n_val]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
