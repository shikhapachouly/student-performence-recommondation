# Manuscript Revision Patches

Each file in this folder is a self-contained text block to paste into the Word manuscript **in red font** (per editor's instruction).

| File | Inserts at | Addresses |
| --- | --- | --- |
| `01_abstract.md` | Replace existing Abstract | R1.5 (50+/43 inconsistency) |
| `02_section2_drawbacks.md` | After §2.1, §2.2, §2.3 | R1.1 (drawbacks of conventional techniques) |
| `03_notation_table.md` | New table at end of §3 (before §4) | R1.2 (notation list) |
| `04_section_3_4_feature_eng_clarification.md` | Replace §3.4 leading paragraph | R1.5, R2.4 |
| `05_section_3_9_recommendation_taxonomy.md` | Replace §3.9 numbers | R1.7 (16/6/19 ≠ 43) |
| `06_section_3_10_leakage_controls.md` | New §3.10 | R2.4 |
| `07_section_4_3_dl_config.md` | New §4.3 | R2.6 (DL training detail) |
| `08_section_5_4_sota_comparison.md` | New §5.4 | R1.4, R2.1 |
| `09_section_5_5_causal_psm.md` | New §5.5 | R2.2 |
| `10_section_5_6_fairness.md` | New §5.6 | R2.5 |
| `11_section_5_5_text_corrections.md` | Replace flagged sentences | R1.8, R1.9, R1.10 |
| `12_section_6_threats_validity_addendum.md` | Append to existing §5.7 | R2.2 reframing |
| `13_appendix_a_replication.md` | New Appendix A | R2.3 |
| `14_table2_corrected.md` | Replace Table 2 | R1.6 |
| `15_table4_consistency.md` | Edit §5.2 transformer F1 range | R1.10 |
| `16_conflicts_and_authorship.md` | After §6, before References | Editor mandate |
| `17_formatting_checklist.md` | Author-side checklist | Editor format guide |

The numerical values for tables in patches 08/09/10 are filled from
`revision/output/tables/*.csv` after running `python -m revision.code.run_all`.
Until those numbers are produced, placeholders are clearly marked `« fill »`.

## Workflow

1. Run `python -m revision.code.run_all` from the project root.
2. Copy the corresponding text from each patch file into the Word manuscript.
3. Apply red font to the inserted/edited text.
4. Re-export the final Word document and save the response letter alongside it.
