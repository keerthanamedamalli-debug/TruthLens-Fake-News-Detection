"""
Data preparation pipeline for TruthLens.
Downloads public datasets from Hugging Face (with robust fallbacks) or creates
a high-quality synthetic dataset of 1,200+ varied news articles if offline.
Prepares balanced 80/10/10 train/val/test splits.
"""

import sys
import os
import json
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is on path
current_dir = Path(__file__).resolve().parent
truthlens_dir = current_dir.parent
sys.path.insert(0, str(truthlens_dir))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TruthLens-Data")

METADATA_FILE = config.DATA_DIR / "dataset_metadata.json"


def generate_synthetic_dataset(num_samples: int = 1200) -> pd.DataFrame:
    """
    Generates a realistic synthetic fake news dataset across multiple domains
    (politics, health, science, technology, finance, sports) with realistic
    stylistic variations, sensational fake patterns, and measured real journalistic prose.
    """
    logger.info("Generating realistic synthetic news dataset (%d samples)...", num_samples)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Building blocks for Real News (measured, objective, cited, sober vocabulary)
    real_templates = [
        # Politics
        (
            "Parliamentary committee convenes to review the new national infrastructure bill. "
            "Lawmakers from both ruling and opposition benches debated budgetary allocations for regional highway projects. "
            "According to the official spokesperson, final voting is scheduled for late next week after public review.",
            "thehindu.com"
        ),
        (
            "Elections commission announces updated guidelines for international observer delegations. "
            "Officials emphasized standard operating protocols regarding verification of biometric voter identification. "
            "The formal notification was published on the gazette website following bilateral stakeholder discussions.",
            "reuters.com"
        ),
        (
            "Bilateral trade talks conclude between ministry delegations with preliminary agreements on customs tariffs. "
            "The joint statement affirmed a shared commitment to resolving non-tariff supply chain barriers by the next quarter.",
            "bbc.com"
        ),
        (
            "Supreme Court bench reserves judgment on constitutional validity of amended arbitration clauses. "
            "Senior counsels representing financial institutions argued that expedited timelines promote economic predictability.",
            "indianexpress.com"
        ),
        (
            "Department of Consumer Affairs mandates standard packaging rules for packaged commodity retailers. "
            "Industry representatives have been granted a six-month transition period to update warehouse inventory labeling.",
            "livemint.com"
        ),

        # Health
        (
            "World Health Organization releases updated clinical management recommendations for viral seasonal influenza. "
            "Surveillance data collected across 48 reference laboratories indicates predominant circulating sub-strains remain susceptible to standard antiviral regimens. "
            "Public health authorities urge annual vaccination for geriatric populations and healthcare personnel.",
            "who.int"
        ),
        (
            "Peer-reviewed study published in the Lancet examines long-term cardiovascular outcomes of Mediterranean dietary patterns. "
            "The ten-year randomized cohort observed statistically significant decreases in systolic arterial pressure among participants receiving cold-pressed olive oil supplements.",
            "nature.com"
        ),
        (
            "National Institute of Virology completes genomic sequencing of regional encephalitis clusters. "
            "Laboratory diagnostic findings confirmed vector-borne transmission via culicine mosquitoes, prompting municipal fogging measures.",
            "ndtv.com"
        ),
        (
            "Clinical trial evaluates efficacy of novel second-generation oral GLP-1 receptor agonists in managing metabolic dysregulation. "
            "Results demonstrated sustained glycemic control with low incidence of severe adverse events over a 52-week monitoring phase.",
            "scientificamerican.com"
        ),
        (
            "Department of Health issues advisory regarding safe water disinfection during monsoon flood season. "
            "Municipal water boards are distributing chlorine purification tablets to prevent waterborne diarrheal outbreaks.",
            "thehindu.com"
        ),

        # Technology & Science
        (
            "Space Research Organisation successfully places remote sensing satellite into sun-synchronous polar orbit. "
            "Telemetry signals acquired by ground station confirmed solar panels deployed nominally within seventeen minutes of launch liftoff.",
            "reuters.com"
        ),
        (
            "Semiconductor consortium breaks ground on new multi-billion dollar commercial fabrication facility. "
            "The advanced manufacturing plant aims to produce 28-nanometer automotive microcontrollers starting in late 2027.",
            "bloomberg.com"
        ),
        (
            "Geological survey maps critical rare earth element deposits along coastal river delta sedimentary basins. "
            "Exploratory borehole samples demonstrated elevated concentrations of neodymium and dysprosium suitable for industrial magnetic applications.",
            "apnews.com"
        ),
        (
            "Major cloud infrastructure provider announces transition to 100% renewable geothermal and wind power across Nordic data center regions. "
            "Grid interconnection agreements will offset annualized carbon emissions by an estimated 450,000 metric tons.",
            "wsj.com"
        ),
        (
            "National cybersecurity center publishes technical advisory outlining recommended cryptographic migration timelines for post-quantum key exchange algorithms. "
            "Enterprise system administrators are advised to audit legacy TLS configurations before mandatory 2028 compliance.",
            "bbc.com"
        ),

        # Finance & Business
        (
            "Central bank monetary policy committee maintains benchmark policy repo rate unchanged at 6.5 percent. "
            "The governor stated inflation expectations remain anchored within the target tolerance corridor, though headline food prices warrant vigilant monitoring.",
            "livemint.com"
        ),
        (
            "Quarterly corporate earnings reports show resilient revenue growth across major domestic software export corporations. "
            "Management guidance highlighted sustained demand for enterprise digital transformation and cloud modernization consulting contracts.",
            "bloomberg.com"
        ),
        (
            "International Monetary Fund revises global economic growth projection upward by 0.2 percentage points for the current fiscal period. "
            "The World Economic Outlook report cited robust consumer spending in emerging Asian economies alongside moderating energy commodities.",
            "reuters.com"
        ),
        (
            "National stock exchange introduces T+0 settlement cycle on an optional basis for select large-cap equity instruments. "
            "Broking houses noted the mechanism enhances capital efficiency while reducing systemic counterparty clearing exposure.",
            "economist.com"
        )
    ]

    # Building blocks for Fake News (sensational, urgent, clickbait, conspiratorial, emotive)
    fake_templates = [
        # Health Hoaxes
        (
            "SHOCKING: Doctors BANNED this miracle herbal tea because it cures all stage 4 cancer in 48 HOURS! "
            "Big Pharma executives are panicking and desperately trying to scrub this secret video off the internet. "
            "Just boiling two kitchen spices dissolves tumors completely with zero side effects! Share this before it gets deleted!",
            "naturalnews.com"
        ),
        (
            "URGENT ALERT: Whistleblower leaks confidential memo showing ordinary tap water contains programmable biological nanobots! "
            "The government doesn't want you to know the real truth behind recent water tower upgrades. "
            "Thousands of people are already feeling the terrifying effects! Drink only raw well water immediately!",
            "beforeitsnews.com"
        ),
        (
            "Scientists are STUNNED: Ancient Himalayan mineral discovered that reverses biological aging by 30 years in just 3 days! "
            "Celebrities have been hiding this youth elixir from the general public for decades. "
            "Click here to claim your emergency bottle before global supplies run out!",
            "yournewswire.com"
        ),
        (
            "BREAKING: Secret laboratory memo confirms 5G towers emit invisible harmonic frequencies that cause instant cardiac arrest! "
            "Hospitals are quietly receiving thousands of patients with melted cell tissues while mainstream news remains totally silent! "
            "You won't believe what happens if you sleep near your WiFi router tonight!",
            "infowars.com"
        ),
        (
            "MIRACLE CURE: Eating raw garlic cloves mixed with warm kerosene completely cures viral respiratory infections overnight! "
            "A courageous independent doctor from village hospital reveals the shocking truth the medical mafia doesn't want you to see!",
            "greatgameindia.com"
        ),

        # Viral India Conspiracies
        (
            "UNESCO OFFICIAL DECLARATION: Indian National Anthem has just been awarded the Best Anthem in the Universe! "
            "All 193 country presidents gave a standing ovation in Paris assembly yesterday! "
            "Every proud citizen must forward this patriotic achievement to at least 10 groups right now!",
            "postcard.news"
        ),
        (
            "ALERT: New 500 and 2000 rupee currency notes have secret satellite tracking microchips embedded under Mahatma Gandhi's glasses! "
            "Tax authorities are using special military radars to detect cash stored in basements up to 120 meters underground! "
            "Exchange all paper money immediately before midnight!",
            "postcard.news"
        ),
        (
            "SPECIAL ANNOUNCEMENT: Government of India is giving free 1,000 rupee mobile recharge and free solar laptops to every student who clicks this link! "
            "Only 500 slots left for today! Forward to your contacts immediately to claim your free reward voucher!",
            "unknown_forward"
        ),
        (
            "WhatsApp cyber police update: From tomorrow morning, all WhatsApp calls will be recorded and chats will show THREE RED TICKS if government investigates you! "
            "Send this emergency notice to all family members so they do not post anti-national jokes!",
            "unknown_forward"
        ),
        (
            "SHOCKING DISCOVERY: NASA satellite photographs reveal holy ancient energy vortexes spinning over Indian sacred mountains during solar eclipse! "
            "Western scientists admitted Vedic texts predicted modern particle physics thousands of years ahead of CERN!",
            "greatgameindia.com"
        ),

        # Politics & Global Hoaxes
        (
            "BOMBSHELL LEAK: Secret United Nations global meeting signs decree abolishing private property ownership by next month! "
            "All residential houses and bank accounts will be seized by global elite committee! "
            "The mainstream corporate media is completely blacking out this terrifying world takeover!",
            "infowars.com"
        ),
        (
            "WORLD SHOCKED: Top government leader secretly arrested and replaced by an artificial clone during international summit! "
            "High-resolution facial scan evidence reveals shocking inconsistencies in earlobe structure! "
            "You will NOT believe who is really running the country behind closed doors!",
            "newswars.com"
        ),
        (
            "BREAKING NEWS: Supreme Court holds secret midnight session and orders immediate arrest of all opposition politicians! "
            "Army divisions surrounding capital city as martial law declared across all major transportation networks!",
            "worldnewsdailyreport.com"
        ),
        (
            "UNBELIEVABLE: Secret Antarctic military wall discovered guarding the edge of flat earth! "
            "Whistleblower pilot risks his life to release banned aerial photographs showing giant ice fortress guarded by laser cannons!",
            "beforeitsnews.com"
        ),

        # Financial Scams
        (
            "GUARANTEED RICHES: Billionaire launches secret automated quantum trading software that turns $100 into $15,000 every single morning! "
            "Central banks tried to arrest him live on air for sharing this legal loophole! Register your bank account before this page is banned!",
            "yournewswire.com"
        ),
        (
            "EMERGENCY WARNING: Central bank programmatically cancelling all personal paper savings notes on Friday! "
            "Banks will freeze ATM machines nationwide and convert all funds into expiring digital tokens! Withdraw your entire cash balance right now!",
            "globalresearch.ca"
        )
    ]

    # Synthesize diverse variations by recombining topic entities, metrics, and adjectives
    real_entities = [
        ("Ministry of Finance", "quarterly economic survey", "fiscal deficit narrowed to target parameters"),
        ("Ministry of Health", "epidemiological bulletin", "immunization coverage surpassed 95 percent threshold"),
        ("Department of Telecommunications", "spectrum allocation framework", "open auction concluded with transparent bidding"),
        ("National Highway Authority", "inter-state expressway corridor", "construction reached 85 percent completion milestone"),
        ("Central Pollution Control Board", "air quality monitoring index", "ambient particulate levels stabilized post rain"),
        ("Bhabha Atomic Research Centre", "isotope radiation facility", "indigenous medical cancer diagnostics supply commenced")
    ]

    fake_sensations = [
        ("EXPOSED: Massive underground conspiracy uncovered by anonymous hacker group!", "infowars.com"),
        ("MIND-BLOWING: Government agency caught hiding alien technology in secret Himalayan tunnel!", "beforeitsnews.com"),
        ("SHOCKING REVELATION: What this doctor discovered in ordinary kitchen salt will terrify you!", "naturalnews.com"),
        ("ALERT: If you received this SMS code do NOT reply or your bank account will be wiped clean in 10 seconds!", "postcard.news"),
        ("THEY CAN'T HIDE THIS ANYMORE: Banned video reveals the horrifying secret behind electric vehicles!", "yournewswire.com"),
        ("UNPRECEDENTED DISASTER: Meteorite containing extraterrestrial virus detected heading directly towards major city!", "worldnewsdailyreport.com")
    ]

    rows = []
    # Half Real, Half Fake
    target_each = num_samples // 2

    # Generate Real samples
    for i in range(target_each):
        base_text, source = real_templates[i % len(real_templates)]
        if i >= len(real_templates):
            # Synthesize variation
            entity, doc, outcome = real_entities[i % len(real_entities)]
            text = f"Official update from {entity}: The latest {doc} confirmed that {outcome}. {base_text}"
        else:
            text = base_text
        rows.append({"text": text, "label": 0, "source": source})

    # Generate Fake samples
    for i in range(target_each):
        base_text, source = fake_templates[i % len(fake_templates)]
        if i >= len(fake_templates):
            sens_intro, sens_src = fake_sensations[i % len(fake_sensations)]
            text = f"{sens_intro} {base_text}"
            source = sens_src
        else:
            text = base_text
        rows.append({"text": text, "label": 1, "source": source})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    return df


def download_hf_dataset(timeout_seconds: int = 15) -> pd.DataFrame:
    """
    Attempts to download a standard public fake news dataset from Hugging Face with a strict timeout.
    Falls back gracefully to the synthetic dataset if unavailable, offline, or slow.
    """
    import concurrent.futures
    candidate_datasets = [
        ("GonzaloA/fake_news", "title", "text", "label"),
        ("mrm8488/fake-news", "title", "text", "label"),
        ("liar", "statement", None, "label")
    ]

    def _try_load(ds_name, col_title, col_text, col_label):
        from datasets import load_dataset
        ds = load_dataset(ds_name, split="train", trust_remote_code=False)
        return ds.to_pandas()

    try:
        from datasets import load_dataset
        for ds_name, col_title, col_text, col_label in candidate_datasets:
            try:
                logger.info("Attempting to load public dataset '%s' from Hugging Face (timeout %ds)...", ds_name, timeout_seconds)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_try_load, ds_name, col_title, col_text, col_label)
                    df = future.result(timeout=timeout_seconds)

                logger.info("Successfully fetched '%s' with %d rows.", ds_name, len(df))

                # Normalize text
                if col_text and col_text in df.columns:
                    if col_title and col_title in df.columns:
                        df["text"] = df[col_title].fillna("") + " - " + df[col_text].fillna("")
                    else:
                        df["text"] = df[col_text].fillna("")
                elif col_title and col_title in df.columns:
                    df["text"] = df[col_title].fillna("")
                else:
                    continue

                # Normalize label (0=real, 1=fake)
                if col_label in df.columns:
                    if ds_name == "liar":
                        df["label"] = df[col_label].apply(lambda x: 1 if x in [0, 1, 2] else 0)
                    else:
                        df["label"] = df[col_label].astype(int)

                if "source" not in df.columns:
                    df["source"] = "hf_dataset"

                df = df[df["text"].str.strip().str.len() > 25]

                real_df = df[df["label"] == 0]
                fake_df = df[df["label"] == 1]
                min_len = min(len(real_df), len(fake_df), 2000)
                if min_len < 100:
                    continue

                balanced_df = pd.concat([
                    real_df.sample(min_len, random_state=config.RANDOM_SEED),
                    fake_df.sample(min_len, random_state=config.RANDOM_SEED)
                ]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

                logger.info("Prepared balanced dataset from '%s': %d rows.", ds_name, len(balanced_df))
                balanced_df.attrs["source_name"] = f"Real dataset ({ds_name})"
                return balanced_df

            except Exception as e:
                logger.warning("Could not load candidate dataset '%s' within timeout: %s", ds_name, e)
                continue

    except Exception as e:
        logger.warning("Hugging Face datasets library error: %s", e)

    logger.info("Falling back to robust offline synthetic news dataset.")
    syn_df = generate_synthetic_dataset(num_samples=1400)
    syn_df.attrs["source_name"] = "Synthetic demo dataset"
    return syn_df


def prepare_data(force: bool = False, use_synthetic_only: bool = False):
    """
    Main data preparation function.
    Loads/generates dataset, creates 80/10/10 splits, and writes metadata.
    """
    if not force and config.TRAIN_DATA_PATH.exists() and config.TEST_DATA_PATH.exists():
        logger.info("Data splits already exist at %s. Skipping preparation.", config.DATA_DIR)
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("Active dataset source: %s", meta.get("data_source", "Unknown"))
            return meta

    if use_synthetic_only:
        logger.info("Explicit request for synthetic dataset.")
        df = generate_synthetic_dataset(num_samples=1400)
        df.attrs["source_name"] = "Synthetic demo dataset"
    else:
        df = download_hf_dataset()

    data_source_name = getattr(df, "attrs", {}).get("source_name", "Synthetic demo dataset")

    # Clean text column
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Split 80 / 10 / 10
    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    # Save to disk
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(config.VAL_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    metadata = {
        "data_source": data_source_name,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "real_count": int((df["label"] == 0).sum()),
        "fake_count": int((df["label"] == 1).sum()),
        "random_seed": config.RANDOM_SEED
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Data preparation complete! Metadata: %s", json.dumps(metadata, indent=2))
    print(f"\n[TruthLens] Active Dataset Source: {data_source_name}")
    print(f"[TruthLens] Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")
    return metadata


def get_dataset_metadata() -> dict:
    """Returns dataset metadata or a default fallback."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"data_source": "Synthetic demo dataset", "total_samples": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data splits for TruthLens")
    parser.add_argument("--synthetic", action="store_true", help="Force generation of synthetic demo dataset")
    parser.add_argument("--force", action="store_true", help="Force recreation of splits")
    args = parser.parse_args()

    prepare_data(force=args.force or args.synthetic, use_synthetic_only=args.synthetic)
