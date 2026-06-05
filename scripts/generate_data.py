# -*- coding: utf-8 -*-
"""Generate the COMMA CSV files from ArgGraph XML files.

This script contains the cleaned data-generation logic for:

- `data/combined_arggraph_dataset.csv` from the XML files under `data/corpus/`;
- `data/neu2.csv` neutral pairs from the combined dataset.

The neutral generation command loads an NLI model and can be slow. The default
arggraph command only uses the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence


FIELDNAMES = ["X", "Y", "Z", "topic", "Source"]


def parse_arggraph(xml_str: str, filename: Optional[str] = None) -> List[Dict[str, str]]:
    """Parse one ArgGraph XML string into pairwise relation rows."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        print(f"Failed to parse XML: {exc}")
        return []

    topic_id = root.get("topic_id", "unknown_topic")
    node_texts: Dict[str, str] = {}

    for edu in root.findall(".//edu"):
        node_id = edu.get("id")
        if node_id is None:
            continue

        text = edu.text.strip() if edu.text else ""
        if text.startswith("<![CDATA[") and text.endswith("]]>"):
            text = text[9:-3].strip()
        node_texts[node_id] = text

    for edge in root.findall('.//edge[@type="seg"]'):
        edu_id = edge.get("src")
        adu_id = edge.get("trg")
        if edu_id and adu_id and edu_id in node_texts:
            node_texts[adu_id] = node_texts[edu_id]

    edge_info: Dict[str, Dict[str, str]] = {}
    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        if edge_id is None:
            continue
        edge_info[edge_id] = {
            "src": edge.get("src", ""),
            "trg": edge.get("trg", ""),
            "type": edge.get("type", ""),
        }

    processed_edges = set()
    dataset: List[Dict[str, str]] = []

    # Add relations that target undercuts are folded into the undercut premise.
    for edge in root.findall('.//edge[@type="add"]'):
        edge_id = edge.get("id")
        src_id = edge.get("src")
        trg_id = edge.get("trg")

        if not all([edge_id, src_id, trg_id]):
            continue

        if trg_id in edge_info and edge_info[trg_id]["type"] == "und":
            und_edge = edge_info[trg_id]
            und_src_id = und_edge["src"]
            und_trg_id = und_edge["trg"]

            if und_trg_id in edge_info:
                target_edge = edge_info[und_trg_id]
                add_text = node_texts.get(src_id, "")
                und_text = node_texts.get(und_src_id, "")
                target_text = node_texts.get(target_edge["src"], "")

                dataset.append(
                    {
                        "X": f"{und_text} {add_text}".strip(),
                        "Y": target_text,
                        "Z": "und",
                        "topic": topic_id,
                        "Source": filename or "unknown",
                    }
                )
                processed_edges.add(edge_id)
                processed_edges.add(trg_id)

    for edge in root.findall(".//edge"):
        edge_id = edge.get("id")
        edge_type = edge.get("type")
        src_id = edge.get("src")
        trg_id = edge.get("trg")

        if not all([edge_id, edge_type, src_id, trg_id]):
            continue
        if edge_id in processed_edges or edge_type == "seg":
            continue

        if edge_type == "und":
            if trg_id in edge_info:
                target_edge = edge_info[trg_id]
                dataset.append(
                    {
                        "X": node_texts.get(src_id, ""),
                        "Y": node_texts.get(target_edge["src"], ""),
                        "Z": edge_type,
                        "topic": topic_id,
                        "Source": filename or "unknown",
                    }
                )
        elif edge_type != "add":
            dataset.append(
                {
                    "X": node_texts.get(src_id, ""),
                    "Y": node_texts.get(trg_id, ""),
                    "Z": edge_type,
                    "topic": topic_id,
                    "Source": filename or "unknown",
                }
            )

    return dataset


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_xml_files(
    folder_path: Path,
    *,
    separate_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Process XML files from a folder and optionally write per-file CSVs."""
    xml_files = sorted(glob.glob(str(folder_path / "*.xml")))
    if not xml_files:
        print(f"No XML files found in {folder_path}")
        return []

    all_data: List[Dict[str, str]] = []

    for xml_file in xml_files:
        xml_path = Path(xml_file)
        print(f"Processing: {xml_path}")
        xml_content = xml_path.read_text(encoding="utf-8")
        data = parse_arggraph(xml_content, xml_path.name)

        if separate_dir is not None:
            write_rows(separate_dir / f"{xml_path.stem}_dataset.csv", data)

        all_data.extend(data)
        print(f"Extracted {len(data)} relations from {xml_path.name}")

    return all_data


def generate_arggraph_dataset(args: argparse.Namespace) -> None:
    """Generate the combined ArgGraph CSV from XML corpus files."""
    rows = process_xml_files(args.input, separate_dir=args.separate_dir)
    if not rows:
        raise SystemExit("No data extracted.")

    write_rows(args.output, rows)
    print(f"Extracted {len(rows)} relations total")
    print(f"Combined dataset saved to {args.output}")


def nli_label(premise: str, hypothesis: str, tokenizer: object, model: object) -> tuple[str, float]:
    """Return the highest-probability NLI label and confidence score."""
    import torch

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(inputs["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    scores = {
        name: round(float(pred) * 100, 1)
        for pred, name in zip(prediction, label_names)
    }
    label = max(scores, key=scores.get)
    return label, scores[label]


def generate_neutral_pairs(args: argparse.Namespace) -> None:
    """Generate candidate neutral pairs using an NLI confidence threshold."""
    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    source_data = pd.read_csv(args.input)

    neutral_rows: List[List[str]] = []
    for _, group in tqdm(source_data.groupby(["Source"])):
        existing_pairs = []
        x_values = []
        y_values = []
        topic = None

        for _, row in group.iterrows():
            x_values.append(row["X"])
            y_values.append(row["Y"])
            existing_pairs.append([row["X"], row["Y"]])
            topic = row["topic"]

        for x_text in x_values:
            for y_text in y_values:
                if x_text == y_text:
                    continue
                if [x_text, y_text] in existing_pairs or [y_text, x_text] in existing_pairs:
                    continue
                candidate = [x_text, y_text, topic, "neutral"]
                if candidate in neutral_rows:
                    continue

                label, score = nli_label(x_text, y_text, tokenizer, model)
                if label == "neutral" and score > args.threshold:
                    neutral_rows.append(candidate)

    pd.DataFrame(neutral_rows).to_csv(args.output)
    print(f"Saved {len(neutral_rows)} neutral rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for data-generation utilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    arggraph = subparsers.add_parser("arggraph", help="generate data/combined_arggraph_dataset.csv")
    arggraph.add_argument("--input", type=Path, default=Path("data/corpus"))
    arggraph.add_argument("--output", type=Path, default=Path("data/combined_arggraph_dataset.csv"))
    arggraph.add_argument("--separate-dir", type=Path, default=None)
    arggraph.set_defaults(func=generate_arggraph_dataset)

    neutral = subparsers.add_parser("neutral", help="generate data/neu2.csv using an NLI model")
    neutral.add_argument("--input", type=Path, default=Path("data/combined_arggraph_dataset.csv"))
    neutral.add_argument("--output", type=Path, default=Path("data/neu2.csv"))
    neutral.add_argument("--threshold", type=float, default=99.0)
    neutral.add_argument(
        "--model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    neutral.set_defaults(func=generate_neutral_pairs)

    return parser


def main() -> None:
    """Run the selected data-generation command."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
