"""Generate a deterministic, public-safe synthetic email corpus.

Stdlib only so anyone can reproduce without API keys.
Run from the repo root:  python data/synthesize.py
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

LABELS = ["personal", "work", "finance", "promo", "transactional", "spam"]

NAMES = ["Alex", "Priya", "Jordan", "Sam", "Mei", "Rafael", "Hana", "Theo", "Yuki", "Nadia"]
COMPANIES = ["Acme Corp", "Northwind", "BlueRiver", "Quanta", "Helio", "Stellaris", "Pinecone Labs"]
PRODUCTS = ["wireless earbuds", "running shoes", "ergonomic chair", "matcha set", "ceramic mug", "linen shirt"]
CITIES = ["Vancouver", "Seattle", "Toronto", "Portland", "Calgary", "Montreal"]
PROJECTS = ["Q3 roadmap", "platform migration", "auth rewrite", "billing v2", "search relevance"]

PERSONAL = [
    ("hey, free this {weekday}?", "Was thinking we grab coffee at {place}. Let me know what works for you."),
    ("photos from {city}", "Finally sorted the photos from our trip. Sharing the album link in a sec."),
    ("{name}'s birthday", "Reminder: {name}'s birthday is on {weekday}. Are we still doing the dinner at {place}?"),
    ("can you help me move?", "Looking for an extra hand on Saturday. Bribing with pizza and good music."),
    ("tomorrow", "Forgot to ask, are you still picking up {name} from the airport tomorrow?"),
    ("housewarming!", "We're finally done unpacking. Come over Saturday around 7. Bring whatever you want to drink."),
    ("re: that book", "Finished it last night. Thoughts: the second half drags but the ending stuck with me."),
    ("checking in", "Hey, haven't heard from you in a while. Things ok?"),
    ("recipe", "Here's the dumpling recipe I promised. The trick is the cold water in the dough."),
    ("plans this weekend", "Free Saturday afternoon? Was thinking the new exhibit at the gallery."),
]

WORK = [
    ("review needed: {project}", "Hi {name}, can you take a pass on the {project} doc by EOD Thursday? Linking the latest draft."),
    ("standup notes", "Quick recap from this morning's standup. Action items below; please flag if I missed anything."),
    ("blocker on {project}", "We're stuck waiting on the schema migration. Pinging {name} to unblock."),
    ("PR open: {project}", "Opened a PR for the {project} change. Should be a small review, mostly config."),
    ("1:1 reschedule", "Hey {name}, can we move our 1:1 to Friday at 2pm? Conflict on my end."),
    ("design doc draft", "First pass at the design doc for {project}. Comments welcome before Monday."),
    ("incident postmortem", "Posting the postmortem for last week's outage. Headlines: rollback worked, alerting did not."),
    ("hiring loop feedback", "Sharing feedback for the {name} loop. Strong technical, want a second opinion on system design."),
    ("quarterly planning", "Setting up planning for next quarter. Block 2 hours on your calendar this Thursday."),
    ("RFC: {project}", "Drafted an RFC for the {project} approach. Looking for input before we commit."),
    ("ship blocker", "We're holding the {project} release pending the security review. ETA tomorrow."),
    ("on-call handoff", "Handing on-call to {name} starting Monday. Runbook updated; ping me if anything is unclear."),
]

FINANCE = [
    ("statement available", "Your {company} credit card statement for this month is now available. Minimum payment due in 21 days."),
    ("low balance alert", "Your checking account balance is below ${amount}. No action required, just a heads up."),
    ("tax document ready", "Your T4A from {company} is available in your account. Download before filing."),
    ("payment received", "We received your loan payment of ${amount}. Next payment is due {weekday} of next month."),
    ("interest rate change", "The variable rate on your line of credit has changed. New rate effective next billing cycle."),
    ("RRSP contribution room", "Your remaining RRSP contribution room for this year is ${amount}."),
    ("e-transfer received", "You received an Interac e-Transfer of ${amount} from {name}."),
    ("investment summary", "Quarterly summary for your portfolio. YTD return below; full breakdown in the attached PDF."),
    ("mortgage renewal", "Your mortgage is up for renewal in 90 days. Rates and options in the attached package."),
    ("fraud alert", "We blocked a suspicious charge of ${amount} on your card. Confirm whether this was you."),
]

PROMO = [
    ("{percent}% off this weekend", "Our biggest sale of the season is here. {percent}% off everything in the {product} collection."),
    ("new arrivals", "Fresh styles just dropped. Take a look at the new {product} line, designed for {city} weather."),
    ("you left something behind", "Still thinking about those {product}? They're in your cart waiting for you."),
    ("members-only preview", "As a {company} member, you get early access to next week's drop. Preview inside."),
    ("free shipping today", "Free shipping on orders over ${amount}, today only. Use code SHIP at checkout."),
    ("survey: $10 reward", "Take a quick survey and we'll send you a $10 credit. Five minutes, promise."),
    ("we miss you", "It's been a while. Here's {percent}% off your next order to welcome you back."),
    ("seasonal lookbook", "Our spring lookbook is live. Inspiration for your {city} weekends."),
    ("flash sale: 24h", "24-hour flash sale on {product}. Stock is moving fast."),
    ("newsletter: weekly digest", "This week in design: five reads, two videos, one tool worth trying."),
]

TRANSACTIONAL = [
    ("order #{order} confirmed", "Thanks for your order. Your {product} will ship within 2 business days. Tracking to follow."),
    ("shipped: order #{order}", "Your order has shipped via {company}. Estimated delivery {weekday}."),
    ("delivered: order #{order}", "Your {product} was delivered today. If anything is wrong, reply within 30 days."),
    ("password reset", "Use this link to reset your password. It expires in 30 minutes."),
    ("login from new device", "We noticed a new sign-in from {city}. If this was you, no action needed."),
    ("verify your email", "Click the link below to verify your email and finish setting up your account."),
    ("appointment confirmed", "Your appointment with Dr. {name} is confirmed for {weekday} at 10am."),
    ("flight check-in open", "Check-in is now open for your flight to {city}. Window closes 1 hour before departure."),
    ("receipt: ${amount}", "Receipt for your purchase at {company}. Total: ${amount}. Thanks for shopping with us."),
    ("subscription renewed", "Your subscription to {company} renewed for ${amount}. Next renewal in 12 months."),
    ("two-factor code", "Your verification code is {code}. It expires in 10 minutes. Don't share this code."),
]

SPAM = [
    ("URGENT: account suspended", "Your account has been suspended due to suspicious activity. Click here to verify your identity within 24 hours."),
    ("you have won ${amount}", "Congratulations! You have been selected as the winner of our weekly draw. Claim your prize here."),
    ("Re: invoice", "Please find attached the overdue invoice. Payment is required immediately to avoid legal action."),
    ("CEO request", "I need you to handle a confidential transaction for me. Reply only to this email. Do not call."),
    ("crypto opportunity", "I am reaching out because you were referred as a serious investor. Returns of 40% guaranteed."),
    ("package undelivered", "Your package could not be delivered. Pay a small customs fee to release it. Click here."),
    ("inheritance notice", "I am a lawyer representing the estate of a distant relative. You may be entitled to a large sum."),
    ("verify your bank", "Unusual activity detected on your account. Confirm your full card number and PIN to restore access."),
    ("love from {city}", "Hello dear, I came across your profile and felt a connection. I am a nurse working overseas."),
    ("limited offer", "Make $5000 a week from home. No experience needed. Reply YES for details."),
]

TEMPLATES = {
    "personal": PERSONAL,
    "work": WORK,
    "finance": FINANCE,
    "promo": PROMO,
    "transactional": TRANSACTIONAL,
    "spam": SPAM,
}


@dataclass
class Email:
    subject: str
    body: str
    label: str


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        name=rng.choice(NAMES),
        company=rng.choice(COMPANIES),
        product=rng.choice(PRODUCTS),
        city=rng.choice(CITIES),
        project=rng.choice(PROJECTS),
        weekday=rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        place=rng.choice(["the cafe", "the new ramen place", "your spot", "the rooftop bar"]),
        amount=rng.choice([10, 25, 50, 100, 250, 500, 1000, 2500]),
        percent=rng.choice([10, 15, 20, 25, 30, 40, 50]),
        order=rng.randint(100000, 999999),
        code=rng.randint(100000, 999999),
    )


def synthesize(n_per_class: int, seed: int) -> list[Email]:
    rng = random.Random(seed)
    out: list[Email] = []
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            subject_t, body_t = rng.choice(templates)
            out.append(Email(subject=fill(subject_t, rng), body=fill(body_t, rng), label=label))
    rng.shuffle(out)
    return out


def write_jsonl(emails: list[Email], path: Path) -> None:
    with path.open("w") as f:
        for e in emails:
            f.write(json.dumps({"subject": e.subject, "body": e.body, "label": e.label}) + "\n")


def main() -> None:
    out_dir = Path(__file__).parent
    splits = {"train": (100, 1), "dev": (25, 2), "test": (34, 3)}
    for name, (n_per_class, seed) in splits.items():
        emails = synthesize(n_per_class=n_per_class, seed=seed)
        write_jsonl(emails, out_dir / f"{name}.jsonl")
        print(f"wrote {len(emails)} emails to {name}.jsonl")


if __name__ == "__main__":
    main()
