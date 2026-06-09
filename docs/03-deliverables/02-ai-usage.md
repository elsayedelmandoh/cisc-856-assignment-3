# AI Usage Disclosure

## CISC 856 Assignment 3 - Implementing and Analyzing TD Algorithms

**elsayed elmandoua - 20596379**

---

## 1. Tool Used

| Tool | Purpose |
|------|---------|
| Deepseek | Assistance and review only, code debugging suggestions, algorithm verification, config checks, report text editing (formatting, tables, structure) |

---

## 2. How GenAI Was Used

### 2.1 Code Assistance (Review and Debugging Only)

| File | Nature of Assistance |
|------|---------------------|
| `src/config/config.py` | I asked GenAI to review it. It suggested using `pydantic.BaseModel` for cleaner type-safe fields. I reviewed the suggestion and applied it. |
| `src/utils/rewards.py` | GenAI pointed out unused parameters and missing imports after I asked for a code check. I cleaned them up. |
| `app.py` | Had a typo (`form` → `from`) and missing references to `Grid`/`Action`/`visualise_policy`. GenAI spotted these when I asked it to verify the file compiled. |
| `notebooks/01-assignment-3.ipynb` | I wrote all cells. GenAI suggested fixing one parameter name, and adding `plt.savefig` calls. I reviewed each change before applying. |

### 2.2 Documentation

| File | Nature of Assistance |
|------|---------------------|
| `requirements.txt` | GenAI noted that `python-dotenv` was unused because it suggested using `pydantic.BaseModel`. I removed it after verifying. |
| `README.md` | I wrote the content. GenAI helped with formatting the project structure tree and results table. |
| `docs/03-deliverables/01-report.pdf` | I wrote the content. GenAI was used only after the fact to clean up Markdown formatting, and structure figure references. |

---

## 3. Declaration

In accordance with Queen's University academic integrity policies and the course policy on gen AI tools:

1. **I wrote and understand all algorithm code**   
GenAI was used only for debugging suggestions, and code review not for writing code
2. **I wrote and own all analysis and ideas in the report**   
GenAI was used only for editing and formatting not for generating content
3. Every GenAI suggestion was reviewed before being applied. Nothing was accepted blindly