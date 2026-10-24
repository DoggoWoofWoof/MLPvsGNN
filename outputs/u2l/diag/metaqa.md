# U2l diagnostic: metaqa / s1sel (decides nothing)

Pick `kt_cos:ada_f12_k0@0.85`. Ledger (U2e's, reproduced): `{"TAU_CUT": 287, "BUDGET": 0, "UNREACHED": 42}`, 57 questions missing a gold.

Questions: `{"questions_missing_a_gold": 57, "by_causes": {"TAU_CUT": 51, "TAU_CUT+UNREACHED": 4, "UNREACHED": 2}, "every_missed_gold_in_a_split_list": 8, "ALL_now": 0.9619, "ALL_if_the_split_lists_were_whole": 0.9673}`

## By cause

### HELD

```json
{
 "golds": 2428,
 "distance": {
  "0": 40,
  "1": 142,
  "2": 566,
  "3": 1680,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 151
 },
 "name_in_question": 0,
 "name_match_B": 15,
 "kinds": {
  "(none)": 2428
 },
 "typed_hop1": {
  "golds": 151,
  "split_list": 48,
  "relation_rank": {
   "n": 151,
   "median": 2.0,
   "p90": 2.0,
   "max": 3.0
  },
  "rank_at_most": {
   "1": 0.4768,
   "3": 1.0,
   "10": 1.0
  },
  "fan_out": {
   "n": 151,
   "median": 1.0,
   "p90": 4.0,
   "max": 13.0
  }
 },
 "typed_hop2": {
  "golds": 597,
  "mid_cvt": 0,
  "any_mid_cvt": 0,
  "split_list": 292,
  "rank1": {
   "n": 597,
   "median": 1.0,
   "p90": 2.0,
   "max": 5.0
  },
  "rank2": {
   "n": 597,
   "median": 1.0,
   "p90": 2.0,
   "max": 3.0
  },
  "rank_both_at_most": {
   "1": 0.4858,
   "3": 0.9966,
   "10": 1.0
  },
  "fan1": {
   "n": 597,
   "median": 23.0,
   "p90": 53.0,
   "max": 166.0
  },
  "fan2": {
   "n": 597,
   "median": 2.0,
   "p90": 5.0,
   "max": 4176.0
  }
 }
}
```

### TAU_CUT

```json
{
 "golds": 287,
 "distance": {
  "0": 0,
  "1": 7,
  "2": 39,
  "3": 241,
  "4": 0,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 7
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 287
 },
 "typed_hop1": {
  "golds": 7,
  "split_list": 5,
  "relation_rank": {
   "n": 7,
   "median": 2.0,
   "p90": 2.0,
   "max": 2.0
  },
  "rank_at_most": {
   "1": 0.2857,
   "3": 1.0,
   "10": 1.0
  },
  "fan_out": {
   "n": 7,
   "median": 5.0,
   "p90": 5.0,
   "max": 5.0
  }
 },
 "typed_hop2": {
  "golds": 39,
  "mid_cvt": 0,
  "any_mid_cvt": 0,
  "split_list": 23,
  "rank1": {
   "n": 39,
   "median": 1.0,
   "p90": 2.0,
   "max": 3.0
  },
  "rank2": {
   "n": 39,
   "median": 1.0,
   "p90": 2.0,
   "max": 4.0
  },
  "rank_both_at_most": {
   "1": 0.4872,
   "3": 0.9744,
   "10": 1.0
  },
  "fan1": {
   "n": 39,
   "median": 11.0,
   "p90": 78.80000000000031,
   "max": 166.0
  },
  "fan2": {
   "n": 39,
   "median": 3.0,
   "p90": 8.0,
   "max": 4176.0
  }
 }
}
```

### UNREACHED

```json
{
 "golds": 42,
 "distance": {
  "0": 0,
  "1": 0,
  "2": 4,
  "3": 34,
  "4": 4,
  "-1": 0
 },
 "one_link_by_family": {
  "structural": 0
 },
 "name_in_question": 0,
 "name_match_B": 0,
 "kinds": {
  "(none)": 42
 },
 "typed_hop1": {
  "golds": 0,
  "split_list": 0,
  "relation_rank": null,
  "rank_at_most": {
   "1": null,
   "3": null,
   "10": null
  },
  "fan_out": null
 },
 "typed_hop2": {
  "golds": 4,
  "mid_cvt": 0,
  "any_mid_cvt": 0,
  "split_list": 2,
  "rank1": {
   "n": 4,
   "median": 1.0,
   "p90": 1.0,
   "max": 1.0
  },
  "rank2": {
   "n": 4,
   "median": 1.0,
   "p90": 2.4000000000000004,
   "max": 3.0
  },
  "rank_both_at_most": {
   "1": 0.75,
   "3": 1.0,
   "10": 1.0
  },
  "fan1": {
   "n": 4,
   "median": 83.5,
   "p90": 166.0,
   "max": 166.0
  },
  "fan2": {
   "n": 4,
   "median": 6.0,
   "p90": 7.0,
   "max": 7.0
  }
 }
}
```

