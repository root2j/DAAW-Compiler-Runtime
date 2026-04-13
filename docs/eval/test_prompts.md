# Test Prompts for Workflow Generation System Evaluation

## Evaluation Dataset — 25 Task Prompts

Each prompt is categorized by **domain**, **complexity** (Simple / Medium / Complex), and **expected characteristics** (whether it should produce linear steps, parallel branches, conditional logic, or loops).

---

### Category 1: Email & Communication Automation (5 prompts)

**P01 | Simple | Linear**
> Log into my Gmail and pull the 10 most recent emails from my inbox. For each one, analyze the sender, subject, and body content. Categorize them as either 'Important' or 'Not Important' based on whether they require my immediate action or contain personal/professional business. Ignore newsletters and automated receipts. Present the results in a clear list.

**P02 | Medium | Conditional Branching**
> Monitor my Gmail inbox every 30 minutes. When a new email arrives from a client (domain: @acmecorp.com), extract any attached invoices as PDF, save them to a Google Drive folder called "Invoices 2026", and send me a Slack notification with the sender name, subject, and attachment filename. If the email has no attachment, just log the subject and sender to a Google Sheet.

**P03 | Medium | Parallel**
> Take a CSV file of 200 customer email addresses and their first names. For each customer, generate a personalized follow-up email using a template that includes their name and references their last purchase. Send the emails via Gmail in batches of 20 with a 5-minute delay between batches. Log the status of each sent email (success/fail) in a spreadsheet.

**P04 | Complex | Conditional + Loop**
> Build a customer support email triage system. Read all unread emails from support@mycompany.com. Classify each email into one of four categories: billing, technical, general inquiry, or complaint. For billing issues, auto-reply with a payment portal link. For technical issues, create a Jira ticket and assign it to the engineering team. For complaints, escalate to a manager via Slack with the full email thread. For general inquiries, draft a response using an FAQ knowledge base and save it as a draft for human review.

**P05 | Simple | Linear**
> Every Monday at 9 AM, pull the top 5 headlines from TechCrunch, summarize each article in 2 sentences, and send the digest to my email address with the subject line "Weekly Tech Digest — [Date]".

---

### Category 2: Data Processing & Analysis (5 prompts)

**P06 | Simple | Linear**
> Read a CSV file called "sales_data.csv" that contains columns for date, product name, quantity, and price. Calculate the total revenue per product, sort by highest revenue, and export the result as a new CSV file called "revenue_summary.csv".

**P07 | Medium | Parallel Branches**
> I have three Excel files — Q1_sales.xlsx, Q2_sales.xlsx, and Q3_sales.xlsx — each with the same column structure. Merge all three into a single dataset, remove any duplicate rows, handle missing values by filling with the column mean, create a pivot table showing monthly revenue by region, and generate a bar chart of the results. Save the final report as a PDF.

**P08 | Complex | Conditional + Loop**
> Connect to a PostgreSQL database and pull all orders from the last 90 days. For each customer with more than 5 orders, calculate their average order value and classify them into tiers: Gold (>$500 avg), Silver ($200-$500), Bronze (<$200). Generate a separate email report for each tier with the customer list, and upload the full analysis as a Google Sheet shared with the marketing team.

**P09 | Medium | Linear**
> Scrape the top 100 product listings from Amazon for the search term "wireless earbuds". Extract the product name, price, rating, and number of reviews. Clean the data by removing duplicates and entries with missing prices. Calculate the average price and rating, and identify the top 10 best-value products (highest rating-to-price ratio). Save to a formatted Excel file.

**P10 | Simple | Linear**
> Read a JSON file containing 500 user records with fields name, email, signup_date, and country. Filter only users who signed up in the last 30 days from India, sort them alphabetically by name, and export the filtered list as both a CSV file and a formatted HTML table.

---

### Category 3: Content Generation & Social Media (5 prompts)

**P11 | Simple | Linear**
> Given a blog post URL, extract the main content, generate a 280-character tweet-sized summary, create 5 relevant hashtags, and format the output as a ready-to-post tweet.

**P12 | Medium | Parallel**
> Take a single product description for a new wireless speaker and generate marketing content for three platforms simultaneously: a 150-word Instagram caption with emoji and hashtags, a professional 300-word LinkedIn post highlighting business use cases, and a casual 100-word Twitter thread (3 tweets). Export all three in a single document with clear section headers.

**P13 | Complex | Loop + Conditional**
> Manage a content calendar for the next 30 days. For each day, check if there is already a scheduled post in the Google Sheet "ContentCalendar". If not, generate a social media post idea relevant to our niche (sustainable fashion), write the caption, suggest an image description for AI generation, and populate the empty slot in the sheet. Skip weekends. At the end, send a summary of all newly created posts to my email.

**P14 | Medium | Linear**
> Take a 2000-word technical blog post about Kubernetes and convert it into: a simplified 500-word version for non-technical readers, a bullet-point executive summary (max 10 bullets), and a set of 5 FAQ questions with answers based on the original content. Save all outputs in a single Markdown file.

**P15 | Simple | Linear**
> Given a company name and its one-line description, generate a 100-word "About Us" section, a tagline (max 10 words), and three value proposition bullet points. Format the output as HTML suitable for embedding in a website.

---

### Category 4: File Management & Document Processing (5 prompts)

**P16 | Simple | Linear**
> Scan a folder called "Downloads" and identify all PDF files larger than 10MB. Compress each one to reduce file size. Move the compressed versions to a new folder called "Compressed_PDFs" and generate a log file listing original filename, original size, new size, and compression percentage.

**P17 | Medium | Conditional**
> Monitor a shared Google Drive folder called "Submissions". When a new file is uploaded, check the file type. If it is a PDF, extract the text and run a plagiarism check against a reference database. If it is a DOCX, convert it to PDF first, then run the same check. Send the plagiarism report to the folder owner via email. If plagiarism exceeds 30%, flag the file by moving it to a subfolder called "Flagged".

**P18 | Complex | Parallel + Conditional**
> Process a batch of 50 scanned receipts (JPEG images) from a folder. For each image, perform OCR to extract the merchant name, date, total amount, and payment method. Validate the extracted data — if any field is missing or ambiguous, flag the receipt for manual review. For valid receipts, categorize expenses into Food, Transport, Office Supplies, and Other. Generate an expense report as a formatted Excel file with a pie chart showing spending distribution, and email the report to finance@mycompany.com.

**P19 | Medium | Loop**
> I have a folder with 200 image files in mixed formats (PNG, JPEG, BMP, TIFF). Convert all of them to WEBP format with 80% quality. Rename each file using the pattern "IMG_YYYY-MM-DD_001.webp" based on the image's EXIF creation date. If no EXIF data exists, use the file's modification date. Move the converted files to a folder called "Processed_Images".

**P20 | Simple | Linear**
> Take a 50-page PDF report and extract all tables from it. Convert each table into a separate sheet in a single Excel workbook. Name each sheet "Table_1", "Table_2", etc. Save the workbook as "extracted_tables.xlsx".

---

### Category 5: API Integration & Multi-Service Workflows (5 prompts)

**P21 | Medium | Linear**
> When a new row is added to a Google Sheet called "Leads", extract the contact name, email, company, and phone number. Create a new contact in HubSpot CRM with these details. Send an automated welcome email via SendGrid using a predefined template. Log the timestamp and status of both operations back to the Google Sheet in new columns.

**P22 | Complex | Parallel + Conditional + Loop**
> Build an automated hiring pipeline. When a new application is submitted via a Google Form, extract the candidate's resume PDF from the form response. Parse the resume to extract name, email, skills, and years of experience. If the candidate has 3+ years of experience and lists Python or Java as a skill, move them to the "Shortlisted" Google Sheet and send them a calendar link for an interview via Calendly. Otherwise, send a polite rejection email. At the end of each day, generate a summary report of all applications received, shortlisted, and rejected, and post it to a Slack channel.

**P23 | Medium | Conditional**
> Integrate a Stripe payment webhook with my system. When a payment succeeds, update the customer's status in my database to "Paid", send a receipt email, and log the transaction in a Google Sheet. When a payment fails, send a reminder email to the customer with a retry link and alert the finance team via Slack.

**P24 | Complex | Parallel + Loop**
> Create a daily competitive intelligence report. Every morning at 7 AM, scrape the latest blog posts from three competitor websites. For each post, generate a one-paragraph summary and a sentiment analysis score. Compare their content themes with our last 10 blog posts. Identify topic gaps where competitors are publishing but we are not. Compile everything into a formatted report and email it to the strategy team. Also post the top 3 findings as a Slack message.

**P25 | Medium | Linear**
> When a new order is placed on my Shopify store, fetch the order details via Shopify API. Generate a shipping label using the ShipStation API. Update the order status to "Shipped" in Shopify. Send the customer a tracking email with the carrier name and tracking number. Log the complete order + shipping details in a Google Sheet.

---

## Evaluation Rubric

For each prompt, evaluate the system output on:

| Metric | Scale | Description |
|--------|-------|-------------|
| **Step Completeness** | 1-5 | Does the output capture all necessary steps from the prompt? |
| **Step Specificity** | 1-5 | Are steps actionable and tool-aware (mentions specific APIs, services) or vague? |
| **Logical Ordering** | 1-5 | Are steps in a correct and logical sequence? |
| **Dependency Awareness** | 1-5 | Does the output recognize which steps depend on others vs. can run in parallel? |
| **Actionability** | 1-5 | Could a developer implement this plan without significant interpretation? |

## Baselines for Comparison

1. **Zero-shot LLM** — Same prompt sent directly to GPT-4 / Claude with: "Break this task into a step-by-step workflow."
2. **CoT LLM** — Same prompt with: "Think step by step about how to automate this task, then provide a detailed workflow."
3. **Your System** — The prompt processed through your multi-agent pipeline.
