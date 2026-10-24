# U2l diagnostic: hotpotqa / s1sel (decides nothing)

Pick `kt_cos:ada_f12_k0@0.85`. Ledger (U2e's, reproduced): `{"TAU_CUT": 13, "BUDGET": 36, "UNREACHED": 20}`, 67 questions missing a gold.

Questions: `{"questions_missing_a_gold": 67, "by_causes": {"BUDGET": 35, "BUDGET+UNREACHED": 1, "TAU_CUT": 13, "UNREACHED": 18}, "every_missed_gold_in_a_split_list": 0, "ALL_now": 0.9556, "ALL_if_the_split_lists_were_whole": 0.9556}`

## By cause

### HELD

```json
{
 "golds": 65,
 "distance": {
  "0": 56,
  "1": 7,
  "2": 2,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 28,
  "ner": 22,
  "knn": 40
 },
 "name_in_question": 29,
 "name_match_B": 16,
 "kinds": {
  "(none)": 65
 }
}
```

### TAU_CUT

```json
{
 "golds": 13,
 "distance": {
  "0": 0,
  "1": 6,
  "2": 7,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 4,
  "ner": 2,
  "knn": 2
 },
 "name_in_question": 1,
 "name_match_B": 0,
 "kinds": {
  "(none)": 13
 }
}
```

### BUDGET

```json
{
 "golds": 36,
 "distance": {
  "0": 0,
  "1": 7,
  "2": 29,
  "3": 0,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 4,
  "ner": 5,
  "knn": 1
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 36
 }
}
```

### UNREACHED

```json
{
 "golds": 20,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 16,
  "3": 3,
  "4": 1,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0,
  "ner": 0,
  "knn": 0
 },
 "name_in_question": 1,
 "name_match_B": 1,
 "kinds": {
  "(none)": 20
 }
}
```

## Links against NER

```json
{
 "w2s_pairs": 32159229,
 "ner_pairs": 20369074,
 "both": 2270050,
 "share_of_w2s_in_ner": 0.0706,
 "share_of_ner_in_w2s": 0.1114,
 "golds_one_link_from_B": {
  "in B": 2489,
  "knn": 12,
  "ner": 15,
  "ner+knn": 10,
  "none": 93,
  "structural": 222,
  "structural+knn": 53,
  "structural+ner": 76,
  "structural+ner+knn": 46
 }
}
```

