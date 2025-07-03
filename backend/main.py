from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

app = FastAPI()
model = YOLO("trained_model.pt")  # Pastikan file ini ada di folder backend/

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Mapping & Utilities ===
RANK_ORDER = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'jack': 11, 'queen': 12, 'king': 13, 'ace': 14
}

ranking_order = [
    "Royal Flush", "Straight Flush", "Four of a Kind", "Full House",
    "Flush", "Straight", "Three of a Kind", "Two Pairs", "One Pair", "High Card"
]

rank_map = {
    '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', 'J': 'jack', 'Q': 'queen', 'K': 'king', 'A': 'ace'
}

def extract_rank_suit(card_label):
    rank_part = card_label[:-1]
    suit_part = card_label[-1].lower()
    rank_name = rank_map.get(rank_part.upper(), rank_part.lower())
    return rank_name, suit_part

def classify_hand(cards_detected):
    cards_detected = list(set(cards_detected))
    if len(cards_detected) < 5:
        return "Not enough cards"

    hand_counter = defaultdict(int)
    best_rank = float('inf')
    best_hand_name = "High Card"

    for combo in combinations(cards_detected, 5):
        ranks = []
        suits = []
        for card in combo:
            rank, suit = extract_rank_suit(card)
            ranks.append(rank)
            suits.append(suit)

        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        rank_values = sorted([RANK_ORDER.get(r, 0) for r in ranks], reverse=True)
        unique_ranks = sorted(set(rank_values), reverse=True)

        is_flush = any(count >= 5 for count in suit_counts.values())
        is_straight = len(unique_ranks) >= 5 and any(
            all(unique_ranks[i + j] - j == unique_ranks[i] for j in range(5))
            for i in range(len(unique_ranks) - 4)
        )
        is_royal = set([10, 11, 12, 13, 14]).issubset(set(rank_values))

        most_common = rank_counts.most_common()

        if most_common[0][1] == 4:
            hand_name = "Four of a Kind"
        elif most_common[0][1] == 3 and len(most_common) > 1 and most_common[1][1] >= 2:
            hand_name = "Full House"
        elif is_flush and is_royal:
            hand_name = "Royal Flush"
        elif is_flush and is_straight:
            hand_name = "Straight Flush"
        elif is_flush:
            hand_name = "Flush"
        elif is_straight:
            hand_name = "Straight"
        elif most_common[0][1] == 3:
            hand_name = "Three of a Kind"
        elif sum(1 for count in rank_counts.values() if count == 2) >= 2:
            hand_name = "Two Pairs"
        elif 2 in rank_counts.values():
            hand_name = "One Pair"
        else:
            hand_name = "High Card"

        hand_counter[hand_name] += 1
        rank_index = ranking_order.index(hand_name)
        if rank_index < best_rank:
            best_rank = rank_index
            best_hand_name = hand_name

    if hand_counter:
        most_frequent = max(hand_counter.items(), key=lambda x: (x[1], -ranking_order.index(x[0])))[0]
        return most_frequent

    return best_hand_name

# === Endpoint ===
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, imgsz=640)[0]
    if results.boxes is None or len(results.boxes.cls) == 0:
        return JSONResponse({"deck": "No cards detected", "cards": []})

    # Ambil label dari bagian atas kartu
    top_labels = {}
    for i, cls in enumerate(results.boxes.cls):
        label = model.names[int(cls)]
        y_top = results.boxes.xyxy[i][1].item()
        if label not in top_labels or y_top < top_labels[label]['y']:
            top_labels[label] = {'label': label, 'y': y_top}
    
    # Urutkan berdasarkan posisi vertikal (opsional agar konsisten)
    detected_cards_info = sorted(top_labels.values(), key=lambda x: x['y'])
    detected_cards = [info['label'] for info in detected_cards_info]

    deck_type = classify_hand(detected_cards)

    return JSONResponse({
        "deck": deck_type,
        "cards": detected_cards
    })