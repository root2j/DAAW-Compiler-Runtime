# DAAW Evaluation Prompts v2

Tests aligned to DAAW's actual tool surface: `web_search`, `http_request`,
`python_exec`, `file_read`, `file_write`, `shell_command`, `notify`.

Every prompt here is **fully executable** — no Gmail/Stripe/Jira dependencies.
Complexity comes from multi-step reasoning, data transformation, and chaining
tool outputs, not from external SaaS integrations.

## Scoring Rubric (1-5 per dimension)

| Dimension | 1 | 5 |
|-----------|---|---|
| **Decomposition** | 1 task, everything collapsed | 3-4+ tasks, each doing one thing |
| **Tool Selection** | Wrong tools or none assigned | Right tool for each step |
| **Dependency Structure** | All independent or all serial | Correct chain with fan-out where appropriate |
| **Output Quality** | Empty / error / hallucinated | Concrete, actionable, matches the ask |
| **End-to-End Success** | Pipeline crashed or all tasks failed | All tasks passed critic |

---

### Category 1: Web Research & Summarization

**P01 | Simple | Linear**
> Search the web for the top 5 programming languages in 2026 by popularity. For each language, find one key advantage and one common criticism. Present the results as a numbered list.

**P02 | Simple | Linear**
> Research the current weather in Tokyo, New York, and London. Compare the temperatures and summarize which city is warmest and which is coldest right now.

**P03 | Medium | Linear**
> Find the 3 most recent SpaceX launches by searching the web. For each launch, note the mission name, date, and whether it was successful. Write the results to a file called "spacex_launches.txt".

**P04 | Medium | Parallel**
> Research the top 3 tourist attractions in both Paris and Rome (6 total). Present them side by side with a brief description of each. Save the comparison to a file called "travel_comparison.md".

---

### Category 2: Data Processing & Computation

**P05 | Simple | Linear**
> Write a Python script that generates a list of the first 20 Fibonacci numbers, then calculate their average. Print the list and the average.

**P06 | Simple | Linear**
> Create a CSV file with 10 sample products (name, price, quantity). Then read it back and calculate the total inventory value (price * quantity for each product, summed).

**P07 | Medium | Linear**
> Write a JSON file containing 5 employee records with name, department, and salary. Then read the file, calculate the average salary, find the highest-paid employee, and write a summary report to "salary_report.txt".

**P08 | Medium | Linear**
> Generate a multiplication table (1-10) using Python, save it to a file, then read the file back and find which product appears most frequently in the table.

---

### Category 3: API Interaction & HTTP

**P09 | Simple | Linear**
> Fetch the list of public APIs from https://api.publicapis.org/entries and count how many APIs are in the "Animals" category. Report the count and list their names.

**P10 | Medium | Linear**
> Fetch a random joke from https://official-joke-api.appspot.com/random_joke, then search the web for the comedian who originated that style of joke. Write both the joke and your research to "joke_research.txt".

**P11 | Medium | Parallel**
> Fetch data from these two public APIs simultaneously: https://catfact.ninja/fact (cat fact) and https://uselessfacts.jsph.pl/api/v2/facts/random (random fact). Compare the lengths of both facts and declare which source gave the longer fact. Save results to "facts_comparison.txt".

**P12 | Medium | Linear**
> Fetch the current Bitcoin price from a public API (search the web for a free Bitcoin price API endpoint first). Then use Python to calculate what 0.5 BTC would be worth. Write the result to "btc_value.txt".

---

### Category 4: File Operations & Text Processing

**P13 | Simple | Linear**
> Create a text file with a 5-paragraph lorem ipsum placeholder. Then read the file, count the total number of words and sentences, and print the statistics.

**P14 | Medium | Linear**
> Write a Python script that generates 50 random names with ages (18-65) as a CSV file. Then read the CSV, group by age decade (teens, twenties, thirties, etc.), and count how many people are in each group. Save the analysis as "age_groups.txt".

**P15 | Medium | Linear**
> Create a JSON configuration file for a hypothetical web application with settings for database host, port, cache TTL, and feature flags. Then read the file, validate that all required fields are present, and generate a documentation page summarizing each setting. Save as "config_docs.md".

---

### Category 5: Multi-Step Research & Analysis

**P16 | Medium | Linear**
> Search the web for the top 3 open-source alternatives to Microsoft Excel. For each, find the latest version number, license type, and one unique feature. Create a markdown comparison table and save it to "spreadsheet_alternatives.md".

**P17 | Complex | Parallel + Linear**
> Research the GDP of the top 5 economies in the world. For each country, also find their population. Use Python to calculate GDP per capita for each. Sort by GDP per capita (highest first) and save the ranked list to "gdp_analysis.txt".

**P18 | Complex | Linear**
> Search the web for the 5 most popular Python web frameworks in 2026. For each framework, find its GitHub star count (search for it). Use Python to create a bar chart description (text-based, not image) ranking them by stars. Save everything to "python_frameworks_report.md".

---

### Category 6: Automation Simulation

**P19 | Simple | Linear**
> Check what the current date and time is using Python. Then search the web for "what happened on this day in history" and write 3 interesting historical events to "today_in_history.txt".

**P20 | Medium | Linear**
> Use Python to generate a random 16-character password containing uppercase, lowercase, digits, and special characters. Then search the web for current password security best practices. Write both the generated password and the security tips to "password_guide.txt".

**P21 | Complex | Linear**
> Search the web for the latest tech news headlines (top 5). For each headline, use the LLM to classify the sentiment (positive, negative, neutral) and the tech sector (AI, hardware, software, crypto, other). Write the classified results as a structured JSON file called "tech_news_analysis.json".
