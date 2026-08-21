# VoiceForge Evaluation Results (Day 10)

Evaluated on 25 held-out test prompts (never seen during SFT or DPO training).

## Rule-based compliance rate
(banned phrase / CTA rule / length target -- % of completions with zero violations)

| Variant | Rule-clean rate |
|---|---|
| BASE | 92% |
| SFT | 100% |
| DPO | 96% |

## Pairwise preference: SFT+DPO vs. SFT-only
(LLM-as-judge, position-randomized, n=25)

- **SFT+DPO win rate: 60%**
- SFT-only wins: 9 (36%)
- Ties: 1

## Rubric scores (1-5, averaged across test set)

| Variant | tone_adherence | vocabulary_compliance | cta_correctness | concrete_detail |
|---|---|---|---|---|
| BASE | 4.08 | 4.84 | 4.72 | 3.16 |
| SFT | 4.80 | 5.00 | 4.88 | 4.08 |
| DPO | 4.48 | 4.92 | 4.72 | 3.40 |

## Qualitative examples

**[romantic_floral] promo_announcement** -- pairwise winner: `sft` (Candidate A fits the 10-20 word length requirement (20 words total) and includes a natural customer story as requested by the brief's angle.)

- BRIEF: [Voice: Romantic Floral] Write a story/promo announcement (10-20 words) for a Valentine's Day bouquet. Campaign angle: customer feature / happy customer story. A light CTA is okay here if it fits naturally, but isn't required. Time-sensitive framing, but not salesy or hypey.
- BASE: Evelyn chose our delicate rose bouquet, each petal telling her love story. Perfect for Valentine’s, delivered in time for that special someone. ❤️✨
- SFT: Sarah sent her mom a pink ribbon bouquet to celebrate 25 years of motherhood. Send us a reference picture to order yours.
- SFT+DPO: Sarah surprised her mom with a handcrafted ribbon bouquet for Valentine's. Send us a reference picture in DM to order yours!

**[cozy_crochet] etsy_listing** -- pairwise winner: `sft` (Candidate B hits all formatting and voice criteria, including the required emoji, concrete detail, restock angle, and proper CTA, whereas Candidate A misses the emoji constraint entirely.)

- BRIEF: [Voice: Cozy Crochet] Write an Etsy-style product listing description (40-70 words, 3-5 sentences) for an amigurumi plushie. Campaign angle: restock of a bestseller. Include a natural CTA encouraging DMs for orders or custom requests. Must include at least one concrete, practical detail (turnaround time, size, material, or care note).
- BASE: Restock your collection with our best-selling 🧶 Snuggly Sleep Bear! 🌸 Soft, fluffy, and perfect for snuggling. Comes in a cozy 8-inch hugger, made from 100% cotton yarn. Crafted with love and patience, expect your cuddle buddy within 3 days! DM us for custom orders or requests. 💛
- SFT: This little handmade plushie is back in stock after a long sell-out run 🧶 Each amigurumi measures a comfy 8 inches tall, making it a great cuddle buddy for your desk or shelf. The plushie is crafted from soft cotton yarn to withstand hugs from all angles, and takes about five days to make from order. Send a DM to start your custom order today!
- SFT+DPO: Your favorite plushie bear is finally back in stock! This little friend is handcrafted with soft, machine-washable cotton yarn and measures about 10 inches tall. Each amigurumi takes between 3 to 5 days to complete. Send me a message to order yours or ask any questions!

**[romantic_floral] etsy_listing** -- pairwise winner: `dpo` (Candidate B falls within the exact word count requirement (51 words, 3 sentences) and includes a concrete, practical detail (3-5 day turnaround) while hitting all persona guidelines. Candidate A is slightly too long (73 words) and exceeds the emoji/formatting limits.)

- BRIEF: [Voice: Romantic Floral] Write an Etsy-style product listing description (40-70 words, 3-5 sentences) for a graduation bouquet special. Campaign angle: new design reveal. Include a natural CTA encouraging DMs for orders or custom requests. Must include at least one concrete, practical detail (turnaround time, size, material, or care note).
- BASE: Graduation is a celebration of new beginnings, and this bespoke bouquet is designed to capture that joy. Each arrangement features a delicate blend of soft pastels and white blooms, wrapped in a luxurious satin ribbon. Ready in just 3 business days, perfect for last-minute celebrations. 🎀 Send us a direct message to order your unique design today or to request custom options!
- SFT: Say goodbye to boring graduation gifts with our exclusive handmade ribbon bouquet 💐 Each stem is made from ultra-soft ribbon that never needs water and lasts forever. Our newest design, the Graduation Ribbon Stem, takes over a week to craft with your choice of ribbon colors. Send us a reference picture in a DM to start your custom order ✨
- SFT+DPO: Meet our newest graduation bouquet design — handcrafted with soft satin ribbon that never needs watering. Each custom order takes around three to five days to create, so mark your calendars ahead! Send us a reference picture in a DM to start your custom order today ✨

**[cozy_crochet] tiktok_caption** -- pairwise winner: `dpo` (Candidate A fits the strict TikTok caption length limit (15-30 words) at 15 words, whereas Candidate B exceeds it at 26 words and includes banned hashtag stacking for this format.)

- BRIEF: [Voice: Cozy Crochet] Write a TikTok caption (15-30 words) for a crochet coaster set. Campaign angle: restock of a bestseller. A light CTA is okay here if it fits naturally, but isn't required.
- BASE: 🌟 Restocked! Our bestselling crochet coaster set is back! 4 coasters, quick delivery. Perfect for any coffee lover! 🧶✨
- SFT: The little details that make our days better 💛 These bestseller coasters are back in stock. DM to order! #handmade #crochet
- SFT+DPO: The best coasters are finally back in stock 🧶 Send me a DM to order yours!

**[romantic_floral] promo_announcement** -- pairwise winner: `dpo` (Candidate B follows the campaign angle of a customer story/feature and stays within the 10-20 word limit without resorting to salesy hype, whereas Candidate A pushes a generic promo message.)

- BRIEF: [Voice: Romantic Floral] Write a story/promo announcement (10-20 words) for an anniversary bouquet special. Campaign angle: customer feature / happy customer story. A light CTA is okay here if it fits naturally, but isn't required. Time-sensitive framing, but not salesy or hypey.
- BASE: Sophie chose our bespoke rose bouquet for her 5th anniversary. Each petal told a love story. 💌 Happy anniversary to you!
- SFT: Meet the perfect blooms to celebrate your forever 💐 A custom anniversary bouquet crafted just for you. DM to order!
- SFT+DPO: Sarah sent her fiancé a custom ribbon bouquet to celebrate five years together.

**[romantic_floral] promo_announcement** -- pairwise winner: `dpo` (Candidate A follows the 10-20 word story/promo announcement length target (11 words) and restricts emojis to the allowed whitelist (1 emoji), whereas Candidate B exceeds the length limit (22 words) and includes a heavy CTA for a format that specifies light or no CTA.)

- BRIEF: [Voice: Romantic Floral] Write a story/promo announcement (10-20 words) for a ribbon bouquet. Campaign angle: restock of a bestseller. A light CTA is okay here if it fits naturally, but isn't required. Time-sensitive framing, but not salesy or hypey.
- BASE: Restocking our bestselling Velvet Romance ribbon bouquet—soft silk roses entwined with love, just in time for your special someone. Order now to send today! ❤️💐
- SFT: The most-requested ribbon bouquet is finally back in stock 🎀 Send us your favorite color palette to order yours today!
- SFT+DPO: Our most-loved ribbon bouquet is back in stock ✨
