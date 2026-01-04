# GADS

## GADS = Generative-augmented Data Science.

### A repo for developing and sharing Data Science tools and methods that utilize generative AI capabilities

This repo is motivated by the observation that the advent of current generative capabilities has had a significant impact on the practice of Data Science in real-life applications, especially, but not exclusively, in the area of NLP. While it is perhaps not surprising that LLMs have revolutionized the field of Natural Language Processing, questions also arise with respect to the effects pre-trained foundation models are having and can have on other aspects of Data Science and Statistical Data Analysis.

In this repo I am exploring what these effects are and how they can be systematically leveraged. The repo is not a software library nor does it attempt to be in any way comprehensive. It is a personal project that others may or may not find interesting.

I am organizing the content to align with the major components of the field of Data Science:

 1. Data Management: This includes all aspects of preparing the data to be analyzed, including Exploratory Data Analysis (EDA)
 2. Modeling: the algorithmic processing of data sets in order to identify and express patterns
 3. Application: the utilization of the fully processed data in order to achieve a desired outcome. This ranges from business strategies being influenced by high-level observations to the production deployment of trained ML models.

I will not proceed in any kind of planned sequence and am most certainly going to be jumping around between topics.

## Google AI Overview

Generative AI enhances classical Data Science by generating synthetic data, automating code/documentation, improving feature engineering, and enabling natural language interfaces for data discovery, while also augmenting traditional models for better predictions (e.g., in fraud detection or medical imaging) by filling data gaps or creating complex data pipelines, ultimately boosting efficiency and innovation in the analytics lifecycle. [1, 2, 3, 4, 5]  
Key Applications 

1. Data Augmentation & Generation: Creates synthetic data (images, text, time-series) to overcome scarcity, especially for rare events (e.g., medical data), improving classical model training. 
2. Feature Engineering & Enrichment: Discovers hidden patterns to suggest new features, enriches datasets with metadata, and automates data preparation (ETL). 
3. Code & Task Automation: Generates Python/SQL code, creates documentation, automates report generation, and assists with debugging, boosting data scientist productivity. 
4. Enhanced Data Discovery: Uses natural language to query data, allowing users to ask questions and get insights without complex coding. 
5. Hybrid Modeling: Combines traditional ML (for prediction) with GenAI (for context/generation), like using GenAI to create realistic scenarios for fraud detection or to provide context around ML predictions. 
6. Model Building: Can assist in designing, training, and evaluating classical ML models by taking instructions and data to build and test them. 
7. Personalization & Business Intelligence: Generates tailored marketing content and assists in creating dynamic visualizations and summaries for faster decision-making. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  

How it Works with Classical ML 

• Bridging Gaps: Generative AI fills gaps where traditional data collection is hard (e.g., rare diseases). 
• Deepening Insights: Extracts complex, unstructured information (like metadata from documents) for structured feature use in classic models. 
• Improving Workflow: Acts as a copilot for data scientists, automating tedious tasks while focusing on core ML problems. [2, 3, 5, 10, 12]  

In essence, GenAI augments, rather than replaces, traditional ML, allowing for broader problem-solving and deeper insights by making data handling more efficient and models more robust. [4, 13]  

AI responses may include mistakes.

[1] https://www.snowflake.com/en/fundamentals/generative-ai-architecture-models-applications/
[2] https://www.usdsi.org/data-science-insights/the-data-science-method-and-generative-ai
[3] https://odsc.medium.com/5-use-cases-for-generative-ai-in-data-analytics-26a91238dbc8
[4] https://www.transorg.ai/blog/why-classical-machine-learning-still-matters-in-a-generative-ai-world/
[5] https://www.researchgate.net/post/How_Can_the_Application_of_Generative_AI_Improve_and_Evolve_Traditional_Machine_Learning_Techniques
[6] https://www.snowflake.com/en/fundamentals/generative-ai/
[7] https://www.analytics8.com/blog/6-use-cases-for-generative-ai/
[8] https://www.aimpointdigital.com/blog/ai-application-planning-choosing-between-traditional-ml-and-generative-ai
[9] https://www.exasol.com/blog/generative-ai-in-data-analytics/
[10] https://medium.com/data-reply-it-datatech/genai-vs-classical-ml-get-the-best-of-both-worlds-3546d3b36528
[11] https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-and-generative-ai-what-are-they-good-for
[12] https://www.youtube.com/watch?v=5BtHm_Sx34U
[13] https://iianalytics.com/community/blog/genai-is-reshaping-data-science-teams