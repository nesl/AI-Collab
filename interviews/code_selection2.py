import pandas as pd
import re
from typing import Union, Iterable, List, Tuple, Dict
import pdb
import argparse

ap = argparse.ArgumentParser(description="Codes")
ap.add_argument("--question", required=True, type=str, help="Question.")

args = ap.parse_args()

# --- Load file ---
df = pd.read_csv("interview_paragraphs_by_session2 - interview_paragraphs_by_session.csv")

# Column names (adjust if yours differ)
participant_col = "Speaker"
question_col = "Question"
codes_col = "Code"


leaders = ["1",  '3', '5',  '7',   '9', '11',  '13', '15', '17',   '19',  '21', '23',   '25', '27',  '29','31']
nonleaders = ['2', '4', '6','8', '10', "12", '14','16','18','20','22','24', '26', '28', '30', '32']

def _norm_set(names):
    return {str(n).strip().lower() for n in names}

def filter_df_by_participants(df_exp: pd.DataFrame, include=None, exclude=None) -> pd.DataFrame:
    """
    Include/exclude participants by name (case-insensitive).
    Use 'include' for keep-only, or 'exclude' to remove.
    You can pass both; 'include' is applied first, then 'exclude'.
    """
    out = df_exp.copy()
    if include is not None:
        inc = _norm_set(include)
        out = out[out["participant"].astype(str).str.lower().isin(inc)]
    if exclude is not None:
        exc = _norm_set(exclude)
        out = out[~out["participant"].astype(str).str.lower().isin(exc)]
    return out

merged_merged = {
    "1": {
        "systematic": ["sensing","information_aggregation","systematic"],
        "first": ["first","second"],
        "clear": ["clear_orders", "strategy_aligned"]
    },
    "2": {
        "last": ["last","third"],
        "orders": ["repeated_order", "cancel"],
        "understanding": ["understanding", "lying"]
    },
    "3": {
        "initiative": ["sense", "carry"],
        "strategy": ["ignored", "strategy", "carrying"],
        "last": ["last", "third"]
        },
    "4": {
        "communication": ["good_communication", "response", "easy_leading"],
        "reponse": ["understanding","repeated_orders", "wording"],
        "curve": ["interface", "learning_curve"],
        "last": ["third","last"]
        },
    "5": {
        "last": ["last","third"],
        "communication": ["fast", "communication"],
        "information": ["information","strategy"],
        "suggestions": ["suggestions", "actuation", "complex"]
        },
    "6": {
        "explanation": ["room", "explanation", "strategy"],
        "response": ["reasoning", "response"]
    },
    "7": {
        "movement": ["come_closer","movement", "visual"]
    },
    "8": {
        "orders": ["orders", "order_follow"]
    },
    "9": {
        "learning": ["learning", "communication"]
    },
    "10": {
        "first": ["first", "second"],
        "last": ["third", "last"],
        "communcation": ["suggestions", "understanding"]
    },
    "11": {
        "first": ["first", "second"],
        "last": ["third", "last"]
    }
}


# --- Your merged categories dictionary, keyed by question key strings ---
merged_categories = {
    "1": {
        "sensing": ["good_object_selection", "sensing_redundancy", "solve_conflicting_estimates", 'accurate_sensing'],
        "information_aggregation": ["information_aggregation"],
        "systematic": ["systematic_behavior", "optimal_planning", "continous_ordering", "realiable_ordering", "ordered_exploration", 'no_mistakes', 'orders_aligned_objective', 'team_efficient', 'finished_task', 'respond_in_time'],
        "first": ["first"],
        "clear_orders": ["clear_orders","understandable_strategy"],
        "strategy_aligned": ["strategy_alignment_with_human", "strategy_aligned_with_humans"],
        "suggestions": ["suggestions_with_reasoning","suggestion_accepted","helpful_suggestions"],
        "second": ["second"],
        "carrying": ['consistent_mover','good_carrying_collaboration', 'objects_carried'],
        'last': ['last'],
        "other": ['following_no_thinking']
    }, 
    "2": {
        "last": ["last"],
        "third": ["third"],
        "repeated_order": ["repeated_orders", "repeated_order"],
        "cancel": ["cancel_without_reason", "changed_orders_abrupt", "changed_orders_abruptly", "changed_orders_fast"],
        "understanding": ["no_understanding", "inconsistency","inconsistent","lack_of_state_understanding", "repeated_orders_needed", "no_response", "disobeying_order", 'carrying_interrupted_by_conversation'],
        "confusion": ["no_strategy_understanding", "confusion_who_performs_task", 'lack_of_direction'],
        "lying": ["lying_task_completion", "false_information", "no_trust_danger", 'no_results_given'],
        "movement": ["blocking_way", "suboptimal_carrying_team", "suboptimal_carrying_execution"],
        "suboptimal": ["carry_heavy_object_far", "no_speed_consideration_plan", "order_sense_empty", "no_multiple_choice", 'no_order_to_carry', 'team_composition', 'suboptimal_behavior'],
        "suggestions": ["ignored_suggestions","suggestions_not_accepted", 'criticize_plan', 'no_need_suggestions'],
        'second': ['second'],
        'first':['first'],
        "other": ['following_no_thinking']
        
    },
    "3": {
        "carry": ['carry_light_objects', 'carry_uncertain_dangerous', 'carried_disagreed_object'],
        "ignored": ['ignored_suggestions', 'ignored_AIs', 'confusing_suggestinons', 'criticize_plan', 'ignore_AIs', 'optimal_plan_human'],
        "orders": ['repeated_orders', 'suggestions_not_accepted', 'no_order_received'],
        "sense": ['move_to_closer_rooms', 'sense_extra', 'scan_different_room', 'complete_previous_order'],
        "interface": ['overwhelming_output', 'difficult_interface', 'unfamiliarity'],
        "third": ['third'],
        "first": ['first'],
        "last": ['last'],
        "carrying": ['suboptimal_carrying_execution', 'human_only_carrying'],
        "strategy": ['agents_slow', 'no_scattering', 'not_carry_heavy'],
        "other": ['no']
    },
    "4": {
        "good_communication": ['easy_communication', 'good_button_communication', 'good_leading_communication_protocol', 'structured_communication', 'good_feedback', 'good_knowing_when_processing', 'good_understanding', 'informative', 'understandable'],
        "wording": ['specific_wording', 'no_shortened_understanding', 'no_shorthand'],
        "interface": ['difficult_interface', 'overwhelming_output', 'overwhelming_responding', 'more_mental_load', 'difficulty_monitoring_task_completion', 'slow_typing'],
        "range": ['difficult_communication_range', 'messages_not_received', 'no_message_acknowledge', 'no_response'],
        "response": ['fast_response'],
        "execution": ['blocking_way', 'delays_execution', 'suboptimal_carrying_execution', 'bad_carrying', 'slow_response', 'stuck_execution', 'suboptimal_carrying_team', 'good_sensing'],
        "understanding": ['no_understanding', 'disobeying_orders', 'not_understanding', 'cancel_without_reason', 'communicated_not_acted', 'conversation_interpreted_as_order', 'false_information', 'forgot_task',  'provide_different_information', 'responded_not_acted','not_accurate', 'suggestions_enabled'],
        "learning_curve": ['learning_curve', 'learning_cuve_communication', 'learning_interface'],
        "last": ['last'],
        "repeated_orders": ['repeated_orders'],
        "easy_leading": ['easier_AI_leading','easy_following'],
        "third": ['third'],
        "other": ['room_not_indicated']
    },
    "5": {
        "complex": ['more_complex_actions'],
        "actuation": ['blocking_way', 'move_faster_carrying', 'bad_carrying'],
        "suggestions": ['more_suggestions', 'change_suboptimal_orders', 'carry_heavy_object_far', 'suggested_while_waiting_order', 'carry_light_objects'],
        "fast": ['fast_ordering', 'overwhelming_output'],
        "last":['last'],
        "communication": ['slow_typing', 'more_button_communication'],
        "leading_better": ['ai_leading_better'],
        "information": ['double_check_object', 'extra_sensing_information', 'get_more_information', 'need_more_sensing'],
        "range": ['greater_communication_range'],
        "bad":['inconsistent', 'misunderstood_order', 'no_response', 'specific_wording'],
        "strategy": ['internal_state_not_known', 'strategy_rationale'],
        "reminders": ['reminders'],
        "repeated_orders": ['repeated_orders'],
        'third': ['third'],
        'sensed_to_review_estimates': ['sensed_to_review_estimates'],
        "other": ['no_thinking_following']
    },
    "6": {
        "filter": ['filter_object_information', 'overwhelming_output', 'fast_responses'],
        "explanation": ['better_explanation', 'feedback_more_specific', 'confused_about_suggestions', "more_perceptual_information"],
        "tasks": ['more_tasks'],
        "room": ['room_information', 'get_state_information'],
        "strategy": ['share_strategy', 'strategy_rationale'],
        "reasoning": ['bad_reasoning', 'no_thinking_following', 'few_sensing_considered', 'repeated_orders'],
        "last": ['last'],
        "execution": ['slow_execution', 'bad_carrying'],
        "response":['no_response'],
        "less":["more_autonomy"]
    },
    "7": {
        "come_closer": ['blocking_way', 'come_closer', 'carrying_heavy_object', 'overwhelming_output'],
        "incomplete": ['incomplete_execution', 'refuse_to_act', 'conversation_interpreted_as_order'],
        "information": ['false_information'],
        "strategy": ['different_strategy', 'unknown_strategy', 'carry_heavy_object_far', 'change_team', 'changing_orders','repeated_orders'],
        "predict": ['easy_to_predict', 'good_task_distribution', 'optimal_planning', 'rational'],
        "movement": ['erratic_movement_patterns', 'slow_execution'],
        "visual": ['visual_limitations'],
        "orders": ["too_much_criticism"],
        "structural":["no_orders_from_non_leaders"]
    },
    "8": {
        "communication": ['structured_communication', 'specific_wording', 'shorten_communication','flexible_communication', 'abstract_communication', 'shortened_communication', 'spam', 'easy_dispel_confusion', "more_communication"],
        "orders": ['delegated_leadership', 'anticipating', 'not_responsive', 'responsive', 'consistent', 'trust'],
        "easier_comms": ['easier_communication','easier_communcation', 'easier_comunication'],
        "agile": ['agile', 'blocking_way', 'constant_communication_follow', 'problems_carrying'],
        "order_follow": ['disobeying_orders', 'no_understanding', 'no_physical_intention_aware', 'need_repeating_orders', 'better_follow_instructions', 'situation_aware', 'selfish'],
        "fast": ['fast_response', 'faster_ordering', 'inconsistent_speed', 'slow_typing', 'slower_ordering'],
        "inflammatory": ['inflammatory'],
        "info": ['information_aggregation', 'more_information', 'location_aware'],
        "learning": ['learning_curve'],
        "moving": ['more_moving', 'optimal_planning', 'different_strategy', 'lack_future_tasking', 'pick_as_discover'],
        "identification": ["more_identifiable"]
    },

    "9": {
        "track": ['keep_track_information', 'monitoring_agents'],
        "learning": ['learning_curve', 'learning_strategy', 'difficult_interface'],
        "communication": ['communication_range_problems'],
        "emotional": ['emotional_orders'],
        "speed": ['different_responding_speed', 'slow_typing'],
        "moving": ['moving_coordination'], 
        "optimal": ['non_optimal_strategy'], 
        "wording": ['specific_wording'],
        "obedience": ["obedience"]
    },
    "10": {
        "last": ['last'],
        "suggestions": ['flexible_suggestions'],
        "third": ['third'],
        "learning": ['learning_curve'],
        "carrying": ['object_carrying', 'human_ai_separation'],
        "first": ['first'],
        "second": ['second'], 
        "information": ['information_aggregation', 'location_aware', "more_context"],
        "optimal": ['optimal_planning', 'optimal_teams', 'order_management','ordering_fast', 'systematic', 'fast_ordering', "faster_processing"],
        "negative": ['tired_human', 'disobeying_orders', 'lack_of_strategy', 'no_response'],
        "understanding": ['human_better_understanding_ai', 'no_misunderstanding']
    },
    "11": {
        "learning": ['learning_curve'],
        "movement": ['movement_stuck', 'slow_execution'],
        "overwhelmed": ['overwhelming_output', 'excessive_suggestions'],
        "second": ['second'],
        "selection": ['bad_selection', 'inefficient_ordering', 'order_management', 'slow_ordering', 'suboptimal_planning', 'unclear_strategy', 'optimal_planning', 'lack_of_strategy'],
        "last": ['last'],
        "misunderstood": ['suggestion_misunderstood', 'disobeying_orders', 'no_response'],
        "first": ['first'],
        "positive": ['human_better_understanding_ai', 'information_aggregation', 'object_carrying', 'ordering_fast'],
        "third": ['third']
    }

    
}


EASIER_TERMS = merged_categories["8"]["easier_comms"]

def parse_q8(text: str) -> Tuple[Union[str, None], List[str]]:
    """
    Q8 special parsing:
      - Accept inputs like "human: yyy, xxx; ai: zzz" or "ai: xxx".
      - Ignore human/ai labels UNLESS one side includes an 'easier_communication' variant.
      - If exactly one side includes that term, set lead_term to that side's label.
      - Aggregate all codes from all segments.
    """
    if not text:
        return (None, [])

    segments = re.split(r"[;|]", str(text))
    all_codes: List[str] = []
    easier_labels = set()

    for seg in segments:
        seg = (seg or "").strip()
        if not seg:
            continue

        # Try to read an explicit label on this segment
        m = re.match(r"^\s*(ai|human)\s*[:\-–]\s*(.*)$", seg, flags=re.IGNORECASE)
        if m:
            label = m.group(1).strip().lower()       # 'ai' or 'human'
            rest  = m.group(2).strip()
            tokens = split_on_commas_pipes(rest)
            all_codes.extend(tokens)
            if any((t or "").strip().lower() in EASIER_TERMS for t in tokens):
                easier_labels.add(label)
        else:
            # No explicit label, just gather tokens
            tokens = split_on_commas_pipes(seg)
            all_codes.extend(tokens)

    lead_term = None
    if len(easier_labels) == 1:
        lead_term = next(iter(easier_labels))

    # Clean codes
    all_codes = [c for c in all_codes if c and str(c).strip()]
    return (lead_term, all_codes)


def build_category_relabel_map(
    merged_categories: Dict[str, Dict[str, List[str]]],
    merged_merged: Dict[str, Dict[str, List[str]]],
    q_key: str,
    case_insensitive: bool = True,
) -> Dict[str, str]:
    """
    For a given question q_key:
      - If merged_merged[q_key][new_label] contains any value that matches an existing
        category name in merged_categories[q_key], those original categories will be
        relabeled to 'new_label'.
      - Any original category not referenced by merged_merged remains as-is.
    Returns a dict: original_category -> final_label
    """
    if q_key not in merged_categories:
        return {}

    orig_cats = list(merged_categories[q_key].keys())
    if case_insensitive:
        lc_orig = {c.lower(): c for c in orig_cats}  # map lower->original casing
    relabel: Dict[str, str] = {}
    used_orig = set()

    # Apply merged_merged mapping if present for this question
    if q_key in merged_merged:
        for new_label, values in merged_merged[q_key].items():
            # Collect all original categories that appear in 'values'
            hits = []
            for v in values:
                if case_insensitive:
                    v_lc = v.lower()
                    if v_lc in lc_orig:
                        hits.append(lc_orig[v_lc])
                else:
                    if v in orig_cats:
                        hits.append(v)

            # Assign the matching original categories to this new_label
            for oc in hits:
                if oc in used_orig:
                    # If one original cat appears under multiple new labels, first wins.
                    # (Change behavior here if you prefer last-wins)
                    continue
                relabel[oc] = new_label
                used_orig.add(oc)

    # Any original categories not mapped → keep their own name
    for oc in orig_cats:
        if oc not in relabel:
            relabel[oc] = oc

    return relabel

def summarize_categories_remapped(
    df_exp: pd.DataFrame,
    q_keys: Union[str, Iterable[str]],
    merged_categories: Dict[str, Dict[str, List[str]]],
    merged_merged: Dict[str, Dict[str, List[str]]],
) -> pd.DataFrame:
    """
    For each question in q_keys:
      - Build relabel map using merged_merged (fallback to original names if not referenced).
      - Compute participants per (final_label), where a participant matches if they have ANY
        code from ANY original category that maps into that final_label.
    Returns columns:
      question_key | category | n_participants | participants | pct_participants | source_categories
    """
    if isinstance(q_keys, str):
        q_keys = [q_keys]

    all_rows = []
    for qk in q_keys:
        qk = str(int(qk))  # normalize "01"->"1"
        if qk not in merged_categories:
            # Nothing to summarize for this question
            continue

        # Filter exploded df to this question
        sub = df_exp[df_exp["q_key"] == qk]
        if sub.empty:
            continue

        # Aggregate per participant: unique set of codes for this question
        per_person = (
            sub.groupby("participant")["codes_list"]
               .apply(lambda s: set(code for codes in s for code in codes))
               .reset_index(name="all_codes")
        )
        total_participants = per_person["participant"].nunique()

        # Build relabel map and reverse index: final_label -> list[original_cats]
        relabel = build_category_relabel_map(merged_categories, merged_merged, qk)
        rev_map: Dict[str, List[str]] = {}
        for orig_cat, final_label in relabel.items():
            rev_map.setdefault(final_label, []).append(orig_cat)

        # For each final_label, union all codes from source original categories
        for final_label, source_cats in rev_map.items():
            # Collect all code tokens associated with these original categories
            all_codes = set()
            for oc in source_cats:
                code_list = merged_categories[qk].get(oc, [])
                all_codes.update(code_list)

            # Find matching participants (ANY code match)
            matched = per_person.loc[
                per_person["all_codes"].apply(lambda codes: len(all_codes.intersection(codes)) > 0),
                "participant"
            ].tolist()

            all_rows.append({
                "question_key": qk,
                "category": final_label,
                "n_participants": len(matched),
                "participants": sorted(matched),
                "pct_participants": round(len(matched) / total_participants * 100,1) if total_participants else 0.0,
                "source_categories": sorted(source_cats),
            })

    if not all_rows:
        return pd.DataFrame(columns=["question_key","category","n_participants","participants","pct_participants","source_categories"])

    result = (pd.DataFrame(all_rows)
                .sort_values(["question_key","n_participants","category"], ascending=[True, False, True])
                .reset_index(drop=True))
    return result


# ---------- Utilities ----------
def parse_question_keys(val) -> List[str]:
    """
    Extract numeric question keys from a cell.
    Handles: '10,11', 'Q10/Q11', '1 & 3', '1–3' ranges, 'Question 2'.
    Returns normalized strings like ['10','11'] (deduped, in order).
    """
    s = str(val).strip().lower()

    keys: List[str] = []
    # Ranges like "1-3" or "10–11"
    for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", s):
        start, end = int(a), int(b)
        keys.extend(str(n) for n in range(min(start, end), max(start, end) + 1))

    # Standalone numbers
    nums = [str(int(x)) for x in re.findall(r"\d+", s)]
    keys.extend(nums)

    # Deduplicate preserving order
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def split_on_commas_pipes(text: str) -> List[str]:
    """Split codes on commas/pipes/semicolons."""
    parts = re.split(r"[,\|;]", text)
    return [p.strip() for p in parts if p and p.strip()]

def split_two_segments(text: str) -> Tuple[str, str]:
    """
    Split a two-part answer into (seg10, seg11) using ';' or '|'.
    If multiple separators exist, split on the first.
    """
    parts = re.split(r"[;|]", text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def parse_segment(segment_text: str) -> Tuple[Union[str, None], List[str]]:
    """
    Parse one segment (e.g., 'ai: last, foo', 'ai', 'human - xxx, yyy', 'ai, xxx').
    Returns (lead_term, codes_list_without_lead).
    """
    raw = (segment_text or "").strip()
    if not raw:
        return (None, [])

    # Accept colon or dash as key/value separator
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*[:\-–]\s*(.*)$", raw)
    if m:
        lead = m.group(1).strip().lower()
        rest = m.group(2).strip()
        codes = split_on_commas_pipes(rest) if rest else []
        return (lead, codes)

    tokens = split_on_commas_pipes(raw)
    if not tokens:
        return (None, [])

    # If first token looks like a leader term, treat it as such
    if tokens[0].lower() in {"ai", "human"}:
        lead = tokens[0].lower()
        return (lead, tokens[1:])

    # Single token → treat as lead
    if len(tokens) == 1:
        return (tokens[0].lower(), [])

    # Otherwise → just codes
    return (None, tokens)

# ---------- Build exploded dataframe with special handling ----------
def build_exploded_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a normalized dataframe with columns:
      - q_key (str), participant (str), lead_term (str|None), codes_list (List[str])
    Rules:
      - All questions exploded from multi-valued cells (e.g., '1,2,3' or '1–3').
      - Q9: strip 'human:'/'ai:'; replace ';' with ','.
      - '10,11' with ';': first segment → Q10; second → Q11.
      - '10,11' without ';' (ambiguous): alternate per participant across such rows: 10,11,10,11,...
      - Single '10'/'11' rows: optional 'word:' prefix supported; ';' treated as ','.
      - All other questions: generic split on commas/pipes/semicolons; no lead_term.
    """
    df2 = df.reset_index(drop=False).rename(columns={"index": "_row_idx"})  # preserve order
    records: List[Dict] = []
    ambiguous_rows: List[Dict] = []   # rows with '10,11' but no ';'

    for _, row in df2.iterrows():
        person = row.get(participant_col)
        q_raw  = row.get(question_col)
        c_raw  = "" if pd.isna(row.get(codes_col)) else str(row.get(codes_col))
        q_keys = parse_question_keys(q_raw)
        if not q_keys:
            continue

        if "10" in q_keys and "11" in q_keys:
            if re.search(r"[;|]", c_raw):
                seg10, seg11 = split_two_segments(c_raw)
                lead10, codes10 = parse_segment(seg10)
                lead11, codes11 = parse_segment(seg11)
                records.extend([
                    {"q_key": "10", "participant": person, "lead_term": lead10, "codes_list": codes10, "_row_idx": row["_row_idx"]},
                    {"q_key": "11", "participant": person, "lead_term": lead11, "codes_list": codes11, "_row_idx": row["_row_idx"]},
                ])
            else:
                ambiguous_rows.append({
                    "_row_idx": row["_row_idx"],
                    "participant": person,
                    "raw_text": c_raw
                })
            continue



        # Handle each q_key individually
        for qk in q_keys:
            text = c_raw

            if qk == "8":
                # Special Q8: labels ignored unless an 'easier_communication' variant appears
                lead_term, codes = parse_q8(text)
                records.append({
                    "q_key": qk,
                    "participant": person,
                    "lead_term": lead_term,   # None unless exactly one side had an 'easier_*' token
                    "codes_list": codes,
                    "_row_idx": row["_row_idx"],
                })
                continue


            if qk == "9":
                # Strip role prefixes and collapse ';' to ','
                text = re.sub(r"\b(?:human|ai)\s*:\s*", "", text, flags=re.IGNORECASE)
                text = text.replace(";", ",")

            if qk in {"10", "11"}:
                text = text.replace(";", ",")
                lead_term, codes = parse_segment(text)
                records.append({
                    "q_key": qk,
                    "participant": person,
                    "lead_term": lead_term,
                    "codes_list": codes,
                    "_row_idx": row["_row_idx"],
                })
            else:
                codes = split_on_commas_pipes(text)
                records.append({
                    "q_key": qk,
                    "participant": person,
                    "lead_term": None,
                    "codes_list": codes,
                    "_row_idx": row["_row_idx"],
                })

    # Resolve ambiguous '10,11' rows (no ';'): alternate per participant
    if ambiguous_rows:
        amb = pd.DataFrame(ambiguous_rows).sort_values("_row_idx", kind="stable")
        amb["_alt_idx"] = amb.groupby("participant").cumcount()
        amb["q_key"] = amb["_alt_idx"].map(lambda i: "10" if (i % 2 == 0) else "11")
        for _, r in amb.iterrows():
            lead, codes = parse_segment(r["raw_text"])
            records.append({
                "q_key": r["q_key"],
                "participant": r["participant"],
                "lead_term": lead,
                "codes_list": codes,
                "_row_idx": r["_row_idx"],
            })

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    # Normalize codes: drop empty
    out["codes_list"] = out["codes_list"].apply(lambda lst: [c for c in lst if c and str(c).strip()])
    return out.sort_values(["_row_idx", "q_key"], kind="stable").reset_index(drop=True)

# ---------- Function 1: Category analysis (includes participant names) ----------
def summarize_categories(df_exp: pd.DataFrame,
                         q_keys: Union[str, Iterable[str]],
                         merged_categories: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """
    For each requested q_key, compute:
      - n_participants with ANY code in each merged category
      - participants (list of names)
      - pct_participants (relative to unique participants in that q_key)
    """
    if isinstance(q_keys, str):
        q_keys = [q_keys]

    rows = []
    for qk in q_keys:
        if qk not in merged_categories:
            continue
        sub = df_exp[df_exp["q_key"] == qk]
        if sub.empty:
            continue

        # Aggregate unique codes per participant for this question
        per_person = (sub.groupby("participant")["codes_list"]
                        .apply(lambda s: set(code for codes in s for code in codes))
                        .reset_index(name="all_codes"))

        for cat, code_list in merged_categories[qk].items():
            matched = per_person.loc[
                per_person["all_codes"].apply(lambda codes: any(c in codes for c in code_list)),
                "participant"
            ].tolist()
            rows.append({
                "question_key": qk,
                "category": cat,
                "n_participants": len(matched),
                "participants": sorted(matched)
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Add totals and percentages; sort
    totals = df_exp.groupby("q_key")["participant"].nunique().rename("n_participants_total")
    result = (result
              .merge(totals, left_on="question_key", right_index=True, how="left"))
    result["pct_participants"] = (result["n_participants"] / result["n_participants_total"] * 100).round(1)
    result = (result
              .sort_values(["question_key", "n_participants", "category"],
                           ascending=[True, False, True])
              .reset_index(drop=True))
    return result

# ---------- Function 2: Lead-term stats (includes participant names) ----------
def lead_term_stats(df_exp: pd.DataFrame,
                    q_keys: Iterable[str] = ("10", "11")) -> pd.DataFrame:
    """
    For selected q_keys (default 10 & 11), compute:
      - n_occurrences of each lead_term (row-level count)
      - n_participants (unique participants per lead_term)
      - participants (sorted unique names)
    """
    q_keys = tuple(str(int(k)) for k in q_keys)  # normalize like "10"
    sub = df_exp[df_exp["q_key"].isin(set(q_keys))].copy()
    sub = sub[pd.notna(sub["lead_term"]) & (sub["lead_term"].astype(str).str.len() > 0)]
    if sub.empty:
        return pd.DataFrame(columns=["question_key", "lead_term", "n_occurrences", "n_participants", "participants"])

    grp = sub.groupby(["q_key", "lead_term"])
    occ = grp.size().rename("n_occurrences")
    parts = grp["participant"].apply(lambda s: sorted(set(s))).rename("participants")
    out = pd.concat([occ, parts], axis=1).reset_index()
    out["n_participants"] = out["participants"].apply(len)

    out = (out.sort_values(["q_key", "n_participants", "n_occurrences", "lead_term"],
                           ascending=[True, False, False, True])
              .reset_index(drop=True))
    return out

# ---------- Run pipeline ----------
'''
def split_codes(val) -> List[str]:
    if pd.isna(val):
        return []
    parts = re.split(r"[,\|;]", str(val))
    return [p.strip() for p in parts if p and p.strip()]
df["codes_list"] = df[codes_col].apply(split_codes)

# Explode so each row is (question, code)
exploded = df[[question_col, "codes_list"]].explode("codes_list").dropna()

# Count frequencies
counts = (
    exploded.groupby([question_col, "codes_list"])
    .size()
    .reset_index(name="count")
    .sort_values([question_col, "count"], ascending=[True, False])
)
pdb.set_trace()
'''

def print_total(df_exp: pd.DataFrame, q_key: str, label: str = ""):
    q_key = str(int(q_key))
    total = df_exp.loc[df_exp["q_key"] == q_key, "participant"].nunique()
    prefix = (label + " ") if label else ""
    print(f"{prefix}TOTAL PARTICIPANTS (Q{q_key}): {total}")
    return total

def print_list(category_list):
    for c in category_list:
        print(c[0],c[1])

df_exp = build_exploded_df(df)

exclude = {"JULIAN"}
df_exp = df_exp[~df_exp["participant"].astype(str).str.lower().isin({p.lower() for p in exclude})]

res_q1 = summarize_categories_remapped(df_exp, args.question, merged_categories, merged_merged)
print_total(df_exp, args.question, label="ALL:")
print(res_q1)

print_list(res_q1[["category", "n_participants"]].values.tolist())
if int(args.question) >= 10 or int(args.question) == 8:
    print(df_exp[df_exp["q_key"] == args.question]["lead_term"].value_counts())
df_filtered = filter_df_by_participants(df_exp, include=leaders)
res_q1 = summarize_categories_remapped(df_filtered, args.question, merged_categories, merged_merged)
print("LEADERS:")
print(res_q1)
print("LEADERS:")
print_list(res_q1[["category", "n_participants"]].values.tolist())
if int(args.question) >= 10 or int(args.question) == 8:
    print(df_exp[(df_exp["q_key"] == args.question) & (df_exp["participant"].isin(leaders))]["lead_term"].value_counts())
df_filtered = filter_df_by_participants(df_exp, include=nonleaders)
res_q1 = summarize_categories_remapped(df_filtered, args.question, merged_categories, merged_merged)
print("NON-LEADERS:")
print(res_q1)
print("NON-LEADERS:")
print_list(res_q1[["category", "n_participants"]].values.tolist())
if int(args.question) >= 10 or int(args.question) == 8:
    print(df_exp[(df_exp["q_key"] == args.question) & (df_exp["participant"].isin(nonleaders))]["lead_term"].value_counts())
#df_exp = filter_df_by_participants(df_exp, include=nonleaders)
res_10_11 = summarize_categories(df_exp, [args.question], merged_categories)
print(res_10_11)

df_filtered = filter_df_by_participants(df_exp, include=nonleaders)
res_10_11 = summarize_categories(df_filtered, [args.question], merged_categories)
print("NON-LEADERS:")
print(res_10_11)

df_filtered = filter_df_by_participants(df_exp, include=leaders)
res_10_11 = summarize_categories(df_filtered, [args.question], merged_categories)
print("LEADERS:")
print(res_10_11)

'''
# Optional: pivot to wide format (questions × codes)
pivot = counts.pivot(index=question_col, columns="codes_list", values="count").fillna(0).astype(int)

# Save
counts.to_csv("codes_by_question_counts.csv", index=False)
pivot.to_csv("codes_by_question_pivot.csv")

#counts[counts["Question"] == "2"]
pdb.set_trace()
print(counts.head())
print("\nWide-format (pivot):\n", pivot.head())

'''

# Multiple questions (once you add "2","3",... to merged_categories)
# res_multi = analyze_questions(df, ["1","2"], merged_categories)
# print(res_multi)
# res_multi.to_csv("category_participant_counts_multi_sorted.csv", index=False)




