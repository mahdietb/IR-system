# FIFA World Cup Information Retrieval System

A Python-based information retrieval system covering FIFA World Cup data from 1930 to 2022, enabling structured querying and analysis across all historical tournament records.

## Overview

This project implements a lightweight IR system over a structured dataset of World Cup matches, teams, scorers, and results. Users can query the data using keyword-based or structured searches, and results are ranked by relevance.

## Features

- Full tournament data from 1930 to 2022 (all 22 editions)
- Match results, group stages, knockout rounds
- Top scorers and team statistics per year
- Query interface for filtering by year, team, player, or stage
- Ranked results based on relevance scoring

## Tech Stack

- **Language:** Python 3
- **Libraries:** (e.g., pandas, re, json — update based on your actual imports)
- **Data format:** CSV / JSON

## Project Structure

```
IR-system/
├── data/
│   └── worldcup_1930_2022.csv   # Main dataset
├── src/
│   ├── indexer.py               # Builds inverted index
│   ├── query_processor.py       # Parses and handles user queries
│   ├── ranker.py                # Scores and ranks results
│   └── main.py                  # CLI entry point
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/mahdietb/IR-system.git
cd IR-system

# Install dependencies
pip install -r requirements.txt

# Run the system
python src/main.py
```

## Example Queries

```
> Which team won in 1998?
> Top scorers in 2014 World Cup
> All matches where Iran played
> Finals results from 1990 to 2022
```

## Dataset

Tournament data sourced from publicly available FIFA World Cup records spanning 1930–2022.

## Author

**Mahdieh Torabi** — [github.com/mahdietb](https://github.com/mahdietb)
