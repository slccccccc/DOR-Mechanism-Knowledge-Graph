# Input Schema

All files are UTF-8 CSV files with a header row. Values should be normalized
names, not patient-level records.

| File | Required columns | Meaning |
| --- | --- | --- |
| `herb_ingredient.csv` | `herb,ingredient` | Herb-to-ingredient edges |
| `herb_symptom.csv` | `herb,symptom` | Herb-to-symptom evidence |
| `ingredient_target.csv` | `ingredient,target` | Ingredient-to-target evidence |

The target file is optional for graph construction, but it is required for
target-level candidate ranking. Duplicate rows are removed automatically.
Private identifiers, clinical records, unredacted review workbooks, and
proprietary source tables must not be committed to the repository.
