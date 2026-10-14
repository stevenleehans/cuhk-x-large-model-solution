"""Canonical action vocabulary.

The dataset names the same action three different ways: HARn path folders
("2_Comb_hair"), HAU option text ("Combing hair") and HARn option text
("combing their hair"). A video model has to speak one language, so everything
maps to a canonical id here. Validated against ground truth in `self_check()`:
every HARn `single` answer must map to the same canonical action as its path.
"""
from __future__ import annotations

FOLDER = {
    "0_Wash_face": "wash_face", "1_Brush_teeth": "brush_teeth",
    "2_Comb_hair": "comb_hair", "3_Take_off_clothes": "undress",
    "4_Wipe_hands": "wipe_hands", "5_Put_on_clothes": "dress",
    "6_Drink_water": "drink", "7_Eat_food": "eat",
    "8_Take_and_use_tableware": "tableware", "9_Pour_drinks": "pour",
    "10_Stir_drinks": "stir", "11_Peel_fruits": "peel_fruit",
    "12_Sweep_the_floor": "sweep", "13_Mop_the_floor": "mop",
    "14_Wipe_bowls": "wipe_bowls", "15_Wipe_windows_and_tables": "wipe_surface",
    "16_Fold_clothes": "fold_clothes", "17_Tap_the_keyboard": "keyboard",
    "18_Write": "write", "19_Make_a_phone_call": "call",
    "20_Check_the_time": "check_time", "21_Read_documents": "read",
    "22_Turn_pages": "turn_page", "23_Listen_to_music_with_headphones": "headphones",
    "24_Use_a_mobile_phone": "use_phone", "25_Watch_TV": "watch_tv",
    "26_Play_games": "games", "27_Take_a_selfie": "selfie",
    "28_Jog_in_place": "jog", "29_Do_squats": "squats",
    "30_Do_jumping_jacks": "jumping_jacks", "31_Do_stretching_exercises": "stretch",
    "32_Stand_up": "stand_up", "33_Lie_down": "lie_down",
    "34_Sit_down": "sit_down", "35_Do_lunges": "lunges",
    "36_Walk": "walk", "37_Take_medicine": "medicine",
    "38_Massage_oneself": "massage", "39_Take_body_temperature": "temperature",
    "40_Stand_on_one_leg_(balance)": "one_leg", "41_Peel_fruits_with_a_knife": "peel_knife",
    "42_Throw_away_(rubbish)": "throw_away",
    "43_Open_the_cabinet_(to_get_things)": "open_cabinet",
}

HAU = {
    "Brushing teeth": "brush_teeth", "Calling": "call",
    "Checking body temperature": "temperature", "Checking the time": "check_time",
    "Combing hair": "comb_hair", "Drinking": "drink", "Eating": "eat",
    "Folding clothes": "fold_clothes", "Getting dressed": "dress",
    "Grabbing utensils": "tableware", "Jumping jacks": "jumping_jacks",
    "Listening to the music with headphones": "headphones", "Lunges": "lunges",
    "Lying down": "lie_down", "Massaging oneself": "massage", "Mopping": "mop",
    "Peeling fruit": "peel_fruit", "Playing games": "games", "Pouring": "pour",
    "Reading": "read", "Running": "jog", "Sitting down": "sit_down",
    "Squats": "squats", "Standing up": "stand_up", "Stirring": "stir",
    "Stretching": "stretch", "Sweeping": "sweep", "Taking a selfie": "selfie",
    "Taking medicine": "medicine", "Turning a page": "turn_page",
    "Typing on a keyboard": "keyboard", "Undressing": "undress",
    "Using a phone": "use_phone", "Walking": "walk",
    "Washing dishes": "wipe_bowls", "Washing face": "wash_face",
    "Watching TV": "watch_tv", "Wiping hands": "wipe_hands",
    "Wiping surface": "wipe_surface", "Writing": "write",
}

HARN_OPTION = {
    "brushing their teeth": "brush_teeth", "checking the time": "check_time",
    "combing their hair": "comb_hair", "doing jumping jacks": "jumping_jacks",
    "doing lunges": "lunges", "doing squats": "squats",
    "doing stretching exercises": "stretch", "drinking water": "drink",
    "eating food": "eat", "folding clothes": "fold_clothes",
    "jogging in place": "jog", "lying down": "lie_down",
    "making a phone call": "call", "massaging themselves": "massage",
    "mopping the floor": "mop", "opening the cabinet": "open_cabinet",
    "peeling fruit": "peel_fruit", "peeling with a knife": "peel_knife",
    "pouring a drink": "pour", "putting on clothes": "dress",
    "putting on headphones": "headphones", "reading documents": "read",
    "sitting down": "sit_down", "standing on one leg": "one_leg",
    "standing up": "stand_up", "stirring a drink": "stir",
    "sweeping the floor": "sweep", "taking a selfie": "selfie",
    "taking medicine": "medicine", "taking off their clothes": "undress",
    "taking their body temperature": "temperature", "throwing away rubbish": "throw_away",
    "turning pages": "turn_page", "typing on the keyboard": "keyboard",
    "using a remote": "watch_tv", "using a smartphone": "use_phone",
    "using tableware": "tableware", "walking": "walk",
    "washing their face": "wash_face", "wiping a bowl": "wipe_bowls",
    "wiping surfaces": "wipe_surface", "wiping their hands": "wipe_hands",
    "writing": "write",
}

# HARn offers 43 option phrases for 44 folders: "using a smartphone" is the
# correct answer for both 24_Use_a_mobile_phone and 26_Play_games. A video model
# scoring HARn options must therefore pool these two canonical actions.
AMBIGUOUS = {"using a smartphone": ("use_phone", "games")}

CANONICAL = sorted(set(FOLDER.values()))
INDEX = {a: i for i, a in enumerate(CANONICAL)}


def canonical_set(text: str) -> tuple[str, ...]:
    """All canonical actions a surface form may denote (see AMBIGUOUS)."""
    lowered = str(text).strip().lower()
    if lowered in AMBIGUOUS:
        return AMBIGUOUS[lowered]
    one = canonical(text)
    return (one,) if one else ()


def canonical(text: str, source: str = "HAU") -> str | None:
    """Map any of the three surface forms to a canonical action id."""
    text = str(text).strip()
    for table in (HAU, HARN_OPTION, FOLDER):
        if text in table:
            return table[text]
    lowered = text.lower()
    return HARN_OPTION.get(lowered) or HAU.get(text.capitalize())
