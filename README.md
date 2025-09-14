# Marketing Mix Model: Understanding Our Revenue Drivers

This project looks at two years of our weekly data to figure out what really drives revenue. The main goal is to build a reliable model that helps us understand the impact of our marketing channels, pricing, and promotions.

A key focus was to solve a common marketing puzzle: how much credit does Google Search really deserve? Is it finding new customers on its own, or is it just "capturing" the interest that our social media ads on Facebook, TikTok, and Snapchat create?

## How to Run This Project

To see the results for yourself, you just need to do three things:

1.  **Clone the repository:** Get a copy of this project on your computer.
2.  **Install the libraries:** Open your terminal, navigate to the project folder, and run the following command. This will install all the tools the project needs.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:** The main analysis is in a Jupyter Notebook. Open the file `notebooks/mmm_analysis.ipynb` and run all the cells from top to bottom.

All the outputs, like charts and data files, will be saved in the `reports/` folder.

## My Approach

I wanted to build a model that was not only accurate but also made sense from a business perspective. Here’s a step-by-step look at how I did it.

### Step 1: Getting the Data Ready

First, I cleaned up the raw data. This involved:

- Making sure dates were handled correctly.
- Filling in any missing values. For marketing spend, I assumed a missing value meant zero dollars were spent. For other metrics like price, I used a smooth interpolation.
- Teaching the model about **seasonality**. I added a few features that help the model understand the natural yearly cycle of sales (e.g., holidays, summer slumps).

I also transformed the marketing spend data to reflect how advertising actually works:

1.  **Carryover Effect (Adstock):** Ads you run this week continue to have an impact for a few weeks to come. I applied a transformation so the model sees this lingering effect.
2.  **Diminishing Returns (Saturation):** Your first $1,000 in ads brings in more sales than your last $1,000. I used a log transformation to help the model understand that spending more and more has less and less impact on the margin.

### Step 2: Solving the "Google Problem" with a Two-Stage Model

This was the most important part. If we just throw all our marketing channels into one big model, it can get confused. It might give Google all the credit for a sale that really started with a customer seeing an ad on TikTok.

To solve this, I used a two-stage approach:

- **Stage 1: The "Why are we spending on Google?" Model.** I first built a small model to predict our _Google spend_ for the week. The main inputs were how much we spent on Facebook, TikTok, and Snapchat. This allowed me to split our Google spend into two parts:

  1.  The part that was a **reaction to our social media ads**.
  2.  The "other" part that was independent of our social efforts.

- **Stage 2: The Final Revenue Model.** With that puzzle solved, I built the main model to predict revenue. But instead of using the simple "Google spend" number, I gave it the two smarter pieces from Stage 1. This lets the model clearly see the impact of social media ads that lead to a Google search, separating it from the impact of Google on its own.

For the model itself, I chose **ElasticNet regression**. It's a great choice here because it's good at handling cases where marketing channels are correlated (e.g., we increase spend everywhere during a big sale) and it helps prevent overfitting.

### Step 3: Making Sure the Model is Reliable

A model that only works on old data is useless. To test it properly, I used a method called **Blocked Time-Series Cross-Validation**. This is a fancy way of saying I always trained the model on the past and tested it on the "future" data it had never seen before. This simulates how we would actually use the model in real life and gives us confidence that it works.

I also checked the model's errors (the "residuals"). I looked at charts to make sure the errors were random and unbiased, which they were. This is a good sign that the model has learned the underlying patterns correctly.

## Key Findings & What We Can Learn

After running the analysis, here are the main takeaways:

1.  **Social Media Creates Demand, Google Captures It:** The model confirms our theory. Social media channels have a strong _indirect_ effect on revenue by driving users to search on Google. This means cutting spend on social would likely also hurt our search performance.

2.  **Price is a Key Lever:** The model is sensitive to price changes. A "sensitivity analysis" showed that a 5% increase in our average price leads to a predictable drop in weekly revenue, and vice-versa.

3.  **Promotions Work:** Running a promotion gives a clear and positive lift to our weekly revenue, even after accounting for all other marketing efforts.

4.  **All Channels Have Diminishing Returns:** The model confirms that simply pumping more money into any single channel isn't a smart strategy. We need a balanced approach.

## What's in the Box? (File Structure)

- `README.md`: This file.
- `notebooks/mmm_analysis.ipynb`: The main notebook with all the steps and code.
- `src/`: A folder with helper Python scripts for data prep, modeling, etc.
- `data/weekly_data.csv`: The dataset used for this analysis.
- `reports/`: Where all the charts, model outputs, and sensitivity reports are saved.
- `requirements.txt`: The list of libraries needed to run the project.
