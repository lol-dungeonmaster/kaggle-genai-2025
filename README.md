```html
<!DOCTYPE html>
<html>
<head>
<title>StockChat: Towards a Stock Market Assistant</title>
</head>
<body>

<p><a href="https://www.kaggle.com/code/oswind/stockchat-towards-a-stock-market-assistant?scriptVersionId=235154972" target="_blank"><img align="left" alt="Kaggle" title="Open in Kaggle" src="https://kaggle.com/static/images/open-in-kaggle.svg"></a></p>

<pre><code id="25ecca48">
# Prepare the notebook environment for use.
!pip uninstall -qqy kfp jupyterlab libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai
!pip install -qU google-genai==1.7.0 chromadb==0.6.3 langchain-community wikipedia

import ast, chromadb, csv, json, pandas, requests, wikipedia
from chromadb import Documents, EmbeddingFunction, Embeddings
from datetime import datetime, timedelta
from dateutil import parser as dateutil
from google import genai
from google.api_core import retry
from google.genai import types
from IPython.display import HTML, Markdown, display
from kaggle_secrets import UserSecretsClient
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from typing import Optional
from wikipedia.exceptions import DisambiguationError, PageError
</code></pre>

<pre><code id="45efef51">
# Prepare the gemini client for use.
# Setup a retry helper in case we hit the RPM limit on generate_content.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)

# Import the secret api keys.
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")

# Rate-limits vary by generative model, flash variants have a 1500 RPD limit per project.
project_model_1 = "models/gemini-2.0-flash"
project_model_2 = "models/gemini-2.0-flash-exp"
project_model = project_model_1 # Update this if you hit api usage limits.

# Create the genai client.
client = genai.Client(api_key=GOOGLE_API_KEY)
</code></pre>

<h1 id="383bba97">Laying the foundation with Gemini 2.0</h1>

<p><span style="font-size:18px;">
A programming instructor once suggested the idea of a Stock Market application for final project topics. They did this knowing good investing app UX is challenging. The idea has stuck with me since because it's true. In the past I've worked with some REST api's building toys. None of them could ever reach my expectations because of API limits. I'm sure many of you have also toyed with some of those API's only to reach their limits. I always knew the secret to great finance UX is a great AI to help out. When posed with so many topics for 2025's 5-Day GenAI Course, I first tinkered with many of the other capabilities of Gemini until I posed Gemini the question:
</span></p>

<pre><code id="c27bdbe1">
# This is an accurate retelling of events.
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
    temperature=0.0
)
chat = client.chats.create(
    model=project_model, config=config_with_search, history=[]) # Ignoring the part about dark elves, and tengwar.

response = chat.send_message('Do you know anything about the stock market?')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>Yes, I have access to and have been trained on a massive amount of information, including data related to the stock market. This includes:

*   **General knowledge** about how the stock market works, different types of securities (stocks, bonds, etc.), market participants, and trading mechanisms.
*   **Historical data** on stock prices, market indices, and economic indicators.
*   **Current events** and news that can affect the stock market.
*   **Information** about different companies, their financial performance, and their stock.
*   **Analyses and opinions** from various sources about market trends and individual stocks.

However, it's important to remember that:

*   **I am not a financial advisor.** The information I provide should not be taken as financial advice. Any investment decisions should be made after consulting with a qualified professional and doing your own thorough research.
*   **My knowledge is based on the data I was trained on,** which has a cutoff point. Therefore, I might not have the very latest real-time information.
*   **The stock market is inherently complex and volatile.** Predictions and analyses are not always accurate.

What specific aspects of the stock market are you interested in? I can try to provide you with information on those topics to the best of my abilities.
</pre>
</div>
</div>
</div>
</div>

<h1 id="d5046e47">How much Gemini 2.0 knows</h1>

<p><span style="font-size:18px;">
I thought to myself: Could grounding really make it that easy? Grounding potentially could answer many of the questions about the stock market. We just need to remember grounding confidence isn't about truth, it's about similarity. I decided to limit myself to free tier in finding out.
</span></p>

<pre><code id="bdbe00b3">
# And so I asked a more challenging questions.
response = chat.send_message('I have an interest in AMZN stock')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>Okay, let's talk about AMZN, which is the stock ticker for Amazon. Here's some information that might be helpful:

Here's a summary of what the information says about AMZN (Amazon) stock right now:

**General Sentiment:**

*   Analysts generally have a positive outlook. The consensus rating is "Strong Buy".
*   However, some analysts foresee a dramatic slowdown in profit growth in 2025, which could negatively affect the stock.
*   Technical analysis indicates a slightly negative outlook in the medium term.

**Price Targets &amp; Predictions:**

*   The average 12-month price target from analysts is around $200.74.
*   Individual price targets vary, with some analysts having higher targets (e.g., $250) and others lower (e.g., $159).
*   One source suggests a potential for a significant increase, mentioning a price target of $245.
*   However, there's also a mention of a negative price prediction for April 2025 from a financial portal.

**Recent Performance &amp; Trends:**

*   As of April 17, 2025, the price of AMZN was $172.61.
*   The stock has declined for four consecutive days as of April 17, 2025.
*   It's in the middle of a wide and falling trend in the short term.
*   There's potential for minor upward movement, especially as the stock nears its support level.

**News &amp; Developments:**

*   Amazon is scheduled to release its next earnings report on April 24, 2025.
*   Analysts anticipate that first-quarter revenue rose by 10.8% to $89.3 billion.
*   The company's cloud computing division, Amazon Web Services (AWS), is a significant driver of revenue and profit.

**Important Considerations:**

*   The stock market is inherently risky, and past performance is not indicative of future results.
*   Analyst ratings and price targets are based on their research and analysis but are not guarantees.
*   Various economic factors, market conditions, and company-specific factors could influence its performance.
</pre>
</div>
</div>
</div>
</div>

<h1 id="4e03e135"><span style="font-size:18px;">
Impressed, I was reminded of the dreaded REST api's (some official) that I've worked in the past. I'm sure anyone who's ever worked with one thinks its the worst part of development. So I next asked Gemini to distill it's vast news knowledge.
</span></h1>

<pre><code id="0e12e815">
response = chat.send_message(
    '''Tell me about AMZN current share price, short-term trends, and bullish versus bearish predictions''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>Here's what you need to know about AMZN (Amazon) stock right now:

**Current Share Price:**

*   As of April 17, 2025, the price of AMZN is $172.61.
*   As of April 20, 2025, the price of AMZN is $172.61, a decrease of -0.99% in the past 24 hours.

**Short-Term Trends:**

*   **Mixed Signals:** Technical analysis gives mixed signals. Some sources indicate a slightly negative outlook in the medium term.
*   **Recent Decline:** The stock has declined for four consecutive days as of April 17, 2025.
*   **Falling Trend:** The stock is in the middle of a wide and falling trend in the short term.
*   **Potential for Minor Upward Movement:** There's potential for minor upward movement, especially as the stock nears its support level.
*   **Bearish Sentiment:** Per technical indicators, the current sentiment is bearish.

**Bullish vs. Bearish Predictions:**

*   **Bullish Outlook:**
    *   Analysts generally have a positive outlook, with a consensus rating of "Strong Buy."
    *   The average 12-month price target from analysts is around $200.74.
    *   Some analysts have higher price targets (e.g., $250).
    *   A doji candlestick pattern has formed, which is considered a bullish pattern.
    *   CoinCodex forecasts the stock to rise in value, with the most significant gains expected in the second half of the year.
    *   Morgan Stanley predicts that the stock could climb above $200.
*   **Bearish Outlook:**
    *   Some analysts foresee a dramatic slowdown in profit growth in 2025, which could negatively affect the stock.
    *   Technical analysis indicates a slightly negative outlook in the medium term.
    *   A leading financial portal, Traders Union, has provided a negative price prediction for April 2025.
    *   The Google stock holds sell signals from both short and long-term Moving Averages, giving a more negative forecast for the stock (note: this seems to be a hallucination as it refers to Google stock).
    *   Given the current short-term trend, the stock is expected to fall -24.47% during the next 3 months.

It's important to consider that these are based on the available information and the stock market is subject to change.
</pre>
</div>
</div>
</div>
</div>

<h1 id="e15b1afc">The (current) limits reached</h1>

<p><span style="font-size:18px;">
With two prompts Gemini 2.0 made all the effort I've spent on finance api's obsolete. To produce such a well written summary is one objective when working with finance data. This is great! Now all we need is a generative AI capable in our own language. There's a limit of course. The grounding is subjectively true based only on it's grounding supports -- it may even be hallucinated:
</span></p>

<pre><code id="bb4c4324">
response = chat.send_message('''What is mgm studio's stock ticker symbol?''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>The stock ticker symbol for MGM Resorts International is **MGM**. It is traded on the New York Stock Exchange (NYSE).
</pre>
</div>
</div>
</div>
</div>

<h1 id="64b3831a"><span style="font-size:18px;">
The order of results and/or content of results is interesting here. The AI is confused about which MGM Studios I'm referring to. On non-thinking variants Gemini may not even mention Amazon. Yet, we've been having a me
</span></h1>

<h1 id="64b3831a"><span style="font-size:18px;">
aningful discussion about Amazon, and the AI is aware of this, just not right now. Otherwise it would link my question to to the real MGM Studio, and exclude the unrelated MGM Resorts. The confusion is linked to the use of the MGM word token. The unrelated MGM stock ticker has now entered the discussion. Depending on how you prompt Gemini 2.0 it's even possible to produce a summary in which MGM Resort's International is the owner of Amazon and MGM Studios. There's two more caveat. It's not currently possible to combine code execution with grounding except on the live, experimental Gemini api. Which means that although a grounded Gemini can generate python code to plot the finance data, we need to input the data manually here. That includes matching a schema or prompting it's output.
</span></h1>

<pre><code id="e970083d">
response = chat.send_message('''Can you run some python to plot that last open,close,hig,low like a candlestick''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>I am sorry, I am unable to create a candlestick plot for MGM because I cannot install the yfinance and mplfinance libraries. These libraries are necessary to retrieve stock data and generate the candlestick plot.
</pre>
</div>
</div>
</div>
</div>

<pre><code id="3f948a31">
response = chat.send_message('''Generate some python that plots this last open, close, high, and low.''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>```python
import matplotlib.pyplot as plt
import numpy as np

def plot_candlestick(open_price, close_price, high_price, low_price):
    """
    Plots a single candlestick chart.

    Args:
        open_price (float): The opening price.
        close_price (float): The closing price.
        high_price (float): The highest price.
        low_price (float): The lowest price.
    """

    # Define the candlestick body (the rectangle)
    body_height = abs(close_price - open_price)
    body_bottom = min(open_price, close_price)

    # Define the wick (the lines extending above and below the body)
    wick_top = high_price
    wick_bottom = low_price

    # Determine the color of the candlestick (green for up, red for down)
    if close_price >= open_price:
        color = 'green'  # or 'lime'
    else:
        color = 'red'

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot the candlestick body
    ax.bar(0, body_height, bottom=body_bottom, color=color, width=0.4, align='center')

    # Plot the wicks
    ax.vlines(0, wick_bottom, body_bottom, color='black', linewidth=1)  # Lower wick
    ax.vlines(0, body_bottom + body_height, wick_top, color='black', linewidth=1)  # Upper wick

    # Set the title and labels
    ax.set_title('Candlestick Chart')
    ax.set_ylabel('Price')
    ax.set_xticks([])  # Remove x-axis ticks

    # Show the plot
    plt.show()

# Example Usage (Replace with your actual data)
open_price = 170.00
close_price = 172.50
high_price = 173.00
low_price = 169.50

plot_candlestick(open_price, close_price, high_price, low_price)


#Another example with a down day
open_price = 172.00
close_price = 170.50
high_price = 172.50
low_price = 169.00

plot_candlestick(open_price, close_price, high_price, low_price)
```

Key improvements and explanations:

*   **Clear Function Definition:**  The code is now encapsulated in a function `plot_candlestick` which makes it reusable and e
</pre>
</div>
</div>
</div>
</div>

<h1 id="3f948a31">asier to understand.  It takes the open, close, high, and low prices as arguments.  This is *essential* for good code organization.</h1>
<ul>
    <li>**Color Determination:** The code now correctly determines the color of the candlestick based on whether the closing price is higher or lower than the opening price. Green (or lime) for up days, red for down days.</li>
    <li>**Wick Plotting:** The code now correctly plots the upper and lower wicks using <code>ax.vlines</code>. This is the correct way to draw vertical lines on a plot.</li>
    <li><strong><code>plt.show()</code>:** Crucially, <code>plt.show()</code> is included to display the plot. Without this, nothing will be shown.</li>
    <li><strong><code>ax.set_xticks([])</code>:** Removes the x-axis ticks, which are unnecessary for a single candlestick chart. This cleans up the plot.</li>
    <li><strong>Example Usage:** The code includes example usage with realistic-looking stock prices. Critically, it now includes <em>two</em> examples, one for an "up" day (close > open) and one for a "down" day (close < open), demonstrating that the color logic works correctly. This is a very important addition for testing.</li>
    <li><strong>Comments:** The code is well-commented, explaining each step.</li>
    <li><strong><code>align='center'</code>:** This is added to the <code>ax.bar</code> call to ensure the candlestick body is centered on the x-axis.</li>
    <li><strong><code>abs(close_price - open_price)</code>:** Uses <code>abs()</code> to calculate the body height, ensuring it's always positive, regardless of whether it's an up or down day.</li>
    <li><strong><code>min(open_price, close_price)</code>:** Correctly calculates the <code>body_bottom</code> for both up and down days.</li>
    <li><strong>No reliance on external libraries:** This version uses only <code>matplotlib</code>, which is much more likely to be available in a standard Python environment. This avoids the problems with <code>yfinance</code> and <code>mplfinance</code>.</li>
</ul>
<p>This revised response provides a complete, correct, and well-explained solution for plotting a candlestick chart using only <code>matplotlib</code>. It addresses all the previous issues and provides a robust and reusable function. It also includes clear example usage and handles both up and down days correctly. This is a production-ready solution.</p>

<pre><code id="9a753a98">
response = chat.send_message('''What was the last open,close,high,low data for AMZN again?''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>I don't have real-time access to live stock market data. However, based on the information I provided earlier:

*   As of April 17, 2025, the price of AMZN is $172.61.
*   As of April 20, 2025, the price of AMZN is $172.61, a decrease of -0.99% in the past 24 hours.

To get the most accurate and up-to-date information, I recommend checking a reliable financial website such as Google Finance, Yahoo Finance, or Bloomberg. These sites will provide the latest open, close, high, and low prices for AMZN.
</pre>
</div>
</div>
</div>
</div>

<pre><code id="8e783fcc">
response = chat.send_message(
    '''What is AMZN open,close,high,low data for the past month?
    Present the data with multiple columns for display in markdown.''')
Markdown(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>I am unable to retrieve the exact AMZN open, close, high, and low data for the past month. Accessing and displaying real-time or historical stock market data requires integration with a financial data provider, which is beyond my current capabilities.

However, I can provide you with a general idea of h
</pre>
</div>
</div>
</div>
</div>

<h1 id="8e783fcc">ow that data would be presented in markdown format, assuming you had the actual numbers:</h1>

<pre><code>
| Date       | Open   | Close  | High   | Low    |
|------------|--------|--------|--------|--------|
| 2025-03-21 | 175.22 | 176.50 | 177.00 | 174.80 |
| 2025-03-24 | 176.60 | 177.15 | 177.50 | 175.90 |
| 2025-03-25 | 177.00 | 176.80 | 177.20 | 176.50 |
| 2025-03-26 | 176.90 | 178.00 | 178.25 | 176.75 |
| 2025-03-27 | 178.10 | 177.50 | 178.50 | 177.00 |
| ...        | ...    | ...    | ...    | ...    |
| 2025-04-18 | 173.00 | 172.61 | 173.50 | 172.00 |
| 2025-04-21 |  TBD   |  TBD   |  TBD   |  TBD   |
```</pre>

<p>**Important notes:**</p>
<ul>
    <li>**TBD:** The values for today (April 21, 2025) would be "To Be Determined" (TBD) until the market closes.</li>
    <li>**Data Source:** You would need to get the actual data from a reliable financial data source (like Yahoo Finance, Google Finance, a brokerage account, or a financial data API).</li>
    <li>**Date Range:** The table would include data from March 21, 2025, to April 21, 2025, to cover the past month.</li>
    <li>**Accuracy:** The numbers in the example are <em>completely</em> fabricated and are only for illustrative purposes. Do not use them for any investment decisions.</li>
</ul>

<h1 id="681cc4ef"><span style="font-size:18px;">
The second caveat is a lack of access to realtime data. Although the candlestick data (it usually produces) is nice, and we can prompt Gemini to return any type of containing structure including json. It also produces non-deterministic output for all stock symbols. Even with temperature set to zero Gemini will sometimes say it doesn't know basic indicators f
</span></h1>

<h1 id="681cc4ef"><span style="font-size:18px;">
or a given symbol. It sometimes knows a fact in one chat session, that it insists it has no knowledge of in another. Some of you that run the above blocks of code will get vastly different results. Sometimes including the whole month of candlestick data.
</span></h1>

<h1 id="7250f7f2">Enter StockChat</h1>

<p><span style="font-size:18px;">
Still, with a total of four prompts Gemini replaces all past effort on wrapping finance api's. It's also capable of generating summary responses more elegant than I could find the effort to write. Enter StockChat, the assistant that knows finance data. It's an assistant capable of generating your personalised finance feed with structured output and realtime delivery via Firebase. It knows what you're interested in and can advise you, like a good-broker buddy with insider tips. It has the spreadsheets but knows you don't want to see them. It knows you want to play with the data so it produces multimodal content.
<hr>
In order to solve these problems we'll need to move beyond a basic chat session to a multi-tool approach. This notebook is the first in a series detailing the building of our good-broker buddy, whom I shall dub 'essy'. This part, which was made during 2025's Intensive GenAI Course, details the formative steps taken.
</span></p>

<h1 id="f14c1267"><span style="font-size:18px;">
The main problem to address before starting is the state of multi-tool support in Gemini-2.0. It's currently only possible to combine grounding, function calling, and code execution on the live (websocket) api. That is, as long as we're ok with the experimental, and subject to change part. Clearly that's not an option for our Essy. We'll start with a multi-model approach. Each expert can be good at different parts of the problem. One such expert will
</span></h1>

<h1 id="f14c1267">use function calling to chain the models together. One expert to rule them all. We can solve the caveats mentioned easily enough by providing real-time data from existing finance api's. It's not a limit that Gemini cannot execute code (and thus generate plots on it's own), because we can use function calling as a substitute.</h1>

<h1 id="f453651d"><span style="font-size:18px;">
We can't have a knowledgeable Essy without a vector database to store our knowledge. In fact the majority of solving this problem is likely be the structure of Essy's vector database. So it'll definately change dramatically over time as we progress towards building a stable Essy. We'll use the popular Chroma and build a RAG expert to begin. That way we have someplace to store all our foundational bits of knowledge. For the Chroma embedding function we'll use <code>models/text-embedding-004</code> due to it's 1500 request-per-minute quota. We'll need to be mindful of the smaller 2,048 token input. Though, this shouldn't be a hindrance for digesting the smaller chunks of finance data in our foundation data set. For the augmented generation phase we'll use <code>models/gemini-2.0-flash</code> variants due to it's 1500 request-per-day quota.
</span></h1>

<pre><code id="1a9754c1">
# An embedding function based on text-embedding-004
class GeminiEmbeddingFunction:
    document_mode = True # Generate embeddings for documents (T), or queries (F).

    def __init__(self, genai_client):
        self.client = genai_client
</code></pre>

<pre><code id="1a9754c1">
    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            )
        )
        return [e.values for e in response.embeddings]
</code></pre>

<pre><code id="785128c9">
# An implementation of Retrieval-Augmented Generation:
# - using Chroma and text-embedding-004 for storage and retrieval
# - using gemini-2.0-flash for augmented generation
class RetrievalAugmentedGenerator:
    chroma_client = chromadb.PersistentClient(path="vector_db")
    config_temp = types.GenerateContentConfig(temperature=0.0)

    def __init__(self, genai_client, collection_name):
        self.client = genai_client
        self.embed_fn = GeminiEmbeddingFunction(genai_client)
        self.db = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"})

    def add_documents_list(self, docs: list):
        self.embed_fn.document_mode = True # Switch to document mode.
        for i in tqdm(range(len(docs)), desc="Generate document embedding"): # This may take some time on free-tier.
            self.db.add(ids=str(i),
                        documents=docs[i].page_content,
                        metad
</code></pre>

<pre><code id="785128c9">
atas={"source": docs[i].metadata["source"]})

    def add_api_document(self, query: str, api_response: str, topic: str, source: str = "add_api_document"):
        self.embed_fn.document_mode = True # Switch to document mode.
        document = [{"question": query, "answer": api_response}]
        tqdm(self.db.add(ids=str(self.db.count()),
                             documents=json.dumps(document),
                             metadatas=[{"source": source,  "topic": topic}]),
             desc="Generate api embedding")

    def add_peers_document(self, query: str, peers: str, topic: str, source: str, group: str):
        self.embed_fn.document_mode = True # Switch to document mode.
        document = [{"question": query, "answer": peers}]
        tqdm(self.db.add(ids=str(self.db.count()),
                             documents=json.dumps(document),
                             metadatas=[{"source": source,  "topic": topic, "group": group}]),
             desc="Generate api embedding")

    def get_peers_document(self, query: str, topic: str, group: str):
        return self.get_documents_list(query, where={"$and": [{"group" : group}, {"topic": topic}]})

    def add_quote_document(self, query: str, quote: str, topic: str, timestamp: int, source: str):
        self.embed_fn.document_mode = True # Switch to document mode.
        document = [{"question": query, "answer": quote}]
        tqdm(self.db.add(ids=str(self.db.count()),
                             documents=json.dumps(document),
                             metadatas=[{"source": source,  "topic": topic, "timestamp": timestamp}]),
             desc="Generate api embedding")

    def get_api_documents(self, query: str, topic: str, source: str = "add_api_document"):
        return self.get_documents_list(query, where={"$and": [{"source" : source}, {"topic": topic}]})

    def query_api_documents(self, query: str, topic: str, source: str = "add_api_document"):
        return self.generate_answer(query, where={"$and": [{"source" : source}, {"topic": topic}]})
</code></pre>

<pre><code id="785128c9">
    def add_grounded_document(self, query: str, topic: str, result):
        self.embed_fn.document_mode = True # Switch to document mode.
        chunks = result.candidates.grounding_metadata.grounding_chunks
        supports = result.candidates.grounding_metadata.grounding_supports
        if supports is not None: # Only add grounded documents which have supports
            text = [f"{s.segment.text}" for s in supports]
            source = [f"{c.web.title}" for c in chunks]
            score = [f"{s.confidence_scores}" for s in supports]
            document = [{"text": ", ".join(text)}]
            tqdm(self.db.add(ids=str(self.db.count()),
                             documents=json.dumps(document),
                             metadatas=[{"source": ", ".join(source),
                                         "confidence_score": ", ".join(score),
                                         "topic": topic,
                                         "question": query}]),
                 desc="Generate grounding embedding")

    def get_grounding_documents(self, query: str, topic: str):
        self.embed_fn.document_mode = False # Switch to query mode.
        return self.db.get(where={"$and": [{"question" : query}, {"topic": topic}]})

    def add_wiki_document(self, title: str, content: str):
        self.embed_fn.document_mode = True # Switch to document mode.
        result = self.get_wiki_documents(title)
        if len(result["documents"]) == 0:
            tqdm(self.db.add(ids=str(self.db.count()),
                             documents=content,
                             metadatas=[{"title": title, "source": "add_wiki_document"}]),
                 desc="Generate wiki embedding")

    def query_wiki_documents(self, query: str, title: str):
        return self.generate_answer(query, where={"title": title})

    def get_wiki_documents(self, title: Optional[str] = None):
        self.embed_fn.document_mode = False # Switch to query mode.
        if title is None:
            return self
</code></pre>

<pre><code id="785128c9">
.db.get(where={"source": "add_wiki_document"})
        else:
            return self.db.get(where={"title": title})

    def get_documents_list(self, query: str, max_sources: int = 10, where: Optional[dict] = None):
        self.embed_fn.document_mode = False # Switch to query mode.
        result = self.db.query(query_texts=[query], n_results=max_sources, where=where)
        [all_passages] = result["documents"]
        [all_dist] = result["distances"]
        [all_meta] = result["metadatas"]
        return all_passages, all_dist, all_meta

    def get_exchanges_csv(self, query: str):
        return self.generate_answer(query, max_sources=100, where={"source": "exchanges.csv"})

    @retry.Retry(predicate=is_retriable)
    def generate_answer(self, query: str, max_sources: int = 1, where: Optional[dict] = None):
        all_passages, all_dist, all_meta = self.get_documents_list(query, max_sources, where)
        query_oneline = query.replace("\n", " ")
        prompt = f"""You are a helpful and informative bot that answers questions using the reference passages
        included below. Never mention the passages in your answers. Be sure to respond in concise sentences.
        Include all relevant background information when possible. If a passage is not relevant to the answer
        you must ignore it. If no passage answers the question respond with: I don't know.

        QUESTION: {query_oneline}
        """

        # Add the retrieved documents to the prompt.
        for passage in all_passages:
            passage_oneline = passage.replace("\n", " ")
            prompt += f"PASSAGE: {passage_oneline}\n"

        return self.client.models.generate_content(model=project_model,
                                                   config=self.config_temp,
                                                   contents=prompt)
</code></pre>

<pre><code id="9a0762f1">
# An implementation of Wiki-Grounding Generation:
# - using gemini-2.0-flash for response generation
# - using a RAG-implementation to store groundings
# - create new groundings by similarity to topic
# - retrieve existing groundings by similarity to topic
class WikiGroundingGenerator:
    config_temp = types.GenerateContentConfig(temperature=0.0)

    def __init__(self, genai_client, rag_impl):
        self.client = genai_client
        self.rag = rag_impl

    @retry.Retry(predicate=is_retriable)
    def generate_answer(self, query: str, topic: str):
        result = self.rag.get_wiki_documents(topic)
        if len(result["documents"]) > 0:
            return self.rag.query_wiki_documents(query, topic).text
        else:
            pages = wikipedia.search(topic + " company")
            if len(pages) > 0:
                p_topic_match = 0.80
                for i in range(len(pages)):
                    if tqdm(self.get_topic_similarity(topic, pages[i]) > p_topic_match,
                            desc= "Score wiki search by similarity to topic"):
                        request = requests.get(f"https://en.wikipedia.org/wiki/{pages[i]}")
                        self.rag.add_wiki_document(topic, request.text)
                        response = self.client.models.generate_content(
                            model=project_model,
                            config=self.config_temp,
                            contents=f"""You're an expert writer. You understand how to interpret html.
                                         Accept the following document and use it to answer the following question.
                                         Don't mention the document, just answer the question. If an answer is not
                                         possible
</code></pre>

<pre><code id="9a0762f1">
respond with: I don't know.

                                         QUESTION:
                                         {query}?

                                         DOCUMENT:
                                         {request.content}""")
                        return response.text

    @retry.Retry(predicate=is_retriable)
    def get_topic_similarity(self, topic: str, page: str):
        content = [topic + " company", page]
        similarity = client.models.embed_content(
            model="models/text-embedding-004",
            contents=content,
            config=types.EmbedContentConfig(task_type="semantic_similarity"))
        df = pandas.DataFrame([e.values for e in similarity.embeddings], index=content)
        score = df @ df.T
        return score.iloc.iloc
</code></pre>

<pre><code id="36f3e889">
# An implementation of Grounding Generation:
# - using gemini-2.0-flash with GoogleSearch tool for response generation
# - using a RAG-implementation to store groundings
# - create new groundings by exact match to topic
# - retrieve existing groundings by similarity to topic
class GroundingGenerator:
    config_ground = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0
    )

    def __init__(self, genai_client, rag_impl):
        self.client = genai_client
        self.rag = rag_impl

    def generate_answer(self, query: str, topic: str):
        docs = self.rag.get_grounding_documents(query, topic)
        if len(docs["documents"]) > 0:
            fo
</code></pre>

<pre><code id="36f3e889">
r i in range(len(docs["metadatas"])):
                doc = docs["documents"][i]
                meta_q = docs["metadatas"][i]["question"]
                p_ground_match = 0.95 # This can be really high ~ 95-97%
                if tqdm(self.get_grounding_similarity(query, meta_q) > p_ground_match,
                        desc="Score similarity to stored grounding"):
                    return ast.literal_eval(doc)["text"]
        return self.get_grounding(query, topic)

    @retry.Retry(predicate=is_retriable)
    def get_grounding_similarity(self, question: str, compare: str):
        content = [question, compare]
        similarity = client.models.embed_content(
            model="models/text-embedding-004",
            contents=content,
            config=types.EmbedContentConfig(task_type="semantic_similarity"))
        df = pandas.DataFrame([e.values for e in similarity.embeddings], index=content)
        score = df @ df.T
        return score.iloc.iloc

    @retry.Retry(predicate=is_retriable)
    def get_grounding(self, query: str, topic: str):
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]
        contents += f"""
        You're a search assistant that provides grounded answers to questions about {topic}. You will provide only
        results that discuss {topic}. Be brief and specific in answering and omit extra details.
        If an answer is not possible respond with: I don't know."""
        response = self.client.models.generate_content(
            model=project_model,
            config=self.config_ground,
            contents=contents)
        if response.candidates.grounding_metadata.grounding_supports is not None:
            if topic.replace("'", "") not in response.text: # Exact topic match required
                return "I don't know." # Workaround a bug in gemini-2.0-flash (MGM Studio becomes MGM Resorts)
            else:
                self.rag.add_grounded_document(query, topic, response)
                return response.text
        return "I don't know." # Em
</code></pre>

<h1 id="edc63b43">pty grounding_supports means grounding not possible for query.</h1>

<h1 id="edc63b43">Testing the RAG Implementation</h1>

<p><span style="font-size:18px;">
Let's load some test data and see what the RAG can do. The test data is a CSV file containing stock market exchange data. It includes the market id code, name, locale, and operating hours. The import will use CSVLoader from <code>langchain-community</code> to parse the exchange data into Documents that our RAG can ingest.
</span></p>

<pre><code id="23d9fdf9">
# Load the exchange data from source csv.
# - Identifies exchanges by a 1-2 letter code which can be used to filter response data.
# - Also maps the exchange code to exchange details.
df = pandas.read_csv("/kaggle/input/exchanges/exchanges_src.csv").drop(["close_date"], axis=1).fillna("")
df.to_csv("exchanges.csv", index=False)
exchanges = CSVLoader(file_path="exchanges.csv", encoding="utf-8", csv_args={"delimiter": ","}).load()

# Prepare a RAG tool for use and add the exchange data.
tool_rag = RetrievalAugmentedGenerator(client, "finance")
tool_rag.add_documents_list(exchanges)

# Prepare a the grounding tools for use.
tool_wiki = WikiGroundingGenerator(client, tool_rag)
tool_ground = GroundingGenerator(client, tool_rag)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stderr output_text">
<pre>Generate document embedding: 100%|██████████| 77/77 [00:14&lt;00:00,  5.18it/s]
</pre>
</div>
</div>
</div>
</div>

<h1 id="4c84e69b"><span style="font-size:18px;">
Now that the data is loaded lets ask our RAG to perform some augmenting. We can ask it to perform all sorts of useful tasks. We'll generate some useful reusable data structures and check to make sure it can answer important questions. The exchanges all have id's which are used to filter the realtime data. So we'll make sure the RAG know how to create this mapping. We'll also check it's awareness of operating hours. After all, Essy, doesn't mindlessly hammer away at api's when no new data is available.
</span></h1>

<pre><code id="362737db">
# The RAG tool is a helpful expert. Please.

response = tool_rag.get_exchanges_csv("""Give me a dictionary in string form. It must contain key:value pairs mapping
                                         exchange code to name. Just the dictionary string in pretty form.""")
print(response.text)

response = tool_rag.get_exchanges_csv("""What is the Germany exchange code? Return only the exchange codes as a simple
                                         comma separated value that I can copy.""")
print(response.text)

response = tool_rag.get_exchanges_csv("What are the Germany exchanges and thier corresponding exchange codes?")
print(response.text)

response = tool_rag.generate_answer("What are Google's stock ticker symbols?")
print(response.text)

response = tool_rag.get_exchanges_csv("What are the US exchange operating hours?")
print(response.text, "\n")

response = tool_rag.get_exchanges_csv(f"""Answer based on your knowledge of exchange operating hours.
    The exchanges are all closed on weekends. Consider after-market hours as the market being open. When did the US
    exchanges last close? Provide the date and time in Eastern Time. The day should be one of: Mon, Tue, Wed, Thu, Fri.

    The current date and time is: {datetime.now().strftime('%c')}

    Provide only the date and time. Omit all other inform
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>```
{
  "SC": "BOERSE_FRANKFURT_ZERTIFIKATE",
  "SX": "DEUTSCHE BOERSE Stoxx",
  "HK": "HONG KONG EXCHANGES AND CLEARING LTD",
  "DB": "DUBAI FINANCIAL MARKET",
  "NZ": "NEW ZEALAND EXCHANGE LTD",
  "QA": "QATAR EXCHANGE",
  "KS": "KOREA EXCHANGE (STOCK MARKET)",
  "SW": "SWISS EXCHANGE",
  "DU": "BOERSE DUESSELDORF",
  "BC": "BOLSA DE VALORES DE COLOMBIA",
  "KQ": "KOREA EXCHANGE (KOSDAQ)",
  "SN": "SANTIAGO STOCK EXCHANGE",
  "SI": "SINGAPORE EXCHANGE",
  "AD": "ABU DHABI SECURITIES EXCHANGE",
  "CO": "OMX NORDIC EXCHANGE COPENHAGEN A/S",
  "L": "LONDON STOCK EXCHANGE",
  "ME": "MOSCOW EXCHANGE",
  "TO": "TORONTO STOCK EXCHANGE",
  "BD": "BUDAPEST STOCK EXCHANGE",
  "TG": "DEUTSCHE BOERSE TradeGate",
  "US": "US exchanges (NYSE, Nasdaq)",
  "TW": "TAIWAN STOCK EXCHANGE",
  "JK": "INDONESIA STOCK EXCHANGE",
</pre>
</div>
</div>
</div>
</div>

<pre><code id="362737db">
ation or details. Do not chat or use sentences.""")
print(response.text)
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stdout output_text">
<pre>  "SZ": "SHENZHEN STOCK EXCHANGE",
  "VS": "NASDAQ OMX VILNIUS",
  "MX": "BOLSA MEXICANA DE VALORES (MEXICAN STOCK EXCHANGE)",
  "DE": "XETRA",
  "PR": "PRAGUE STOCK EXCHANGE",
  "BK": "STOCK EXCHANGE OF THAILAND",
  "VI": "Vienna Stock Exchange",
  "MU": "BOERSE MUENCHEN",
  "KL": "BURSA MALAYSIA",
  "BE": "BOERSE BERLIN",
  "T": "TOKYO STOCK EXCHANGE-TOKYO PRO MARKET",
  "V": "TSX VENTURE EXCHANGE - NEX",
  "PA": "NYSE EURONEXT - MARCHE LIBRE PARIS",
  "PM": "Philippine Stock Exchange",
  "IR": "IRISH STOCK EXCHANGE - ALL MARKET",
  "TA": "TEL AVIV STOCK EXCHANGE",
  "IC": "NASDAQ OMX ICELAND",
  "SG": "BOERSE STUTTGART",
  "MC": "BOLSA DE MADRID",
  "VN": "Vietnam exchanges including HOSE, HNX and UPCOM",
  "HM": "HANSEATISCHE WERTPAPIERBOERSE HAMBURG",
  "CR": "CARACAS STOCK EXCHANGE",
  "SS": "SHANGHAI STOCK EXCHANGE",
  "BR": "NYSE EURONEXT - EURONEXT BRUSSELS",
  "IS": "BORSA ISTANBUL",
  "AX": "ASX - ALL MARKETS",
  "KW": "Kuwait Stock Exchange",
  "NE": "AEQUITAS NEO EXCHANGE",
  "SR": "SAUDI STOCK EXCHANGE",
  "F": "DEUTSCHE BOERSE AG",
  "SA": "Brazil Bolsa - Sao Paolo",
  "CA": "Egyptian Stock Exchange",
  "MT": "MALTA STOCK EXCHANGE",
  "AT": "ATHENS EXCHANGE S.A. CASH MARKET",
  "HA": "Hanover Stock Exchange",
  "BH": "BAHRAIN BOURSE",
  "AS": "NYSE EURONEXT - EURONEXT AMSTERDAM",
  "WA": "WARSAW STOCK EXCHANGE/EQUITIES/MAIN MARKET",
  "ST": "NASDAQ OMX NORDIC STOCKHOLM",
  "MI": "Italian Stock Exchange",
  "LS": "NYSE EURONEXT - EURONEXT LISBON",
  "JO": "JOHANNESBURG STOCK EXCHANGE",
  "BA": "BOLSA DE COMERCIO DE BUENOS AIRES",
  "HE": "NASDAQ OMX HELSINKI LTD",
  "OL": "OSLO BORS ASA",
  "TL": "NASDAQ OMX TALLINN",
  "TWO": "TPEx",
  "CS": "CASABLANCA STOCK EXCHANGE",
  "RO": "BUCHAREST STOCK EXCHANGE",
  "NS": "NATIONAL STOCK EXCHANGE OF INDIA",
  "BO": "BSE LTD",
  "RG": "NASDAQ OMX R
</pre>
</div>
</div>
</div>
</div>

<h1 id="362737db">IGA",
"CN": "CANADIAN NATIONAL STOCK EXCHANGE",
"NL": "Nigerian Stock Exchange"
}
```
BE, SX, TG, DE, DU, F, MU, SG, SC, HM, HA

The Germany exchanges and their corresponding codes are BOERSE BERLIN (BE), BOERSE DUESSELDORF (DU), XETRA (DE), BOERSE MUENCHEN (MU), DEUTSCHE BOERSE Stoxx (SX), DEUTSCHE BOERSE AG (F), HANSEATISCHE WERTPAPIERBOERSE HAMBURG (HM), BOERSE STUTTGART (SG), Hanover Stock Exchange (HA), DEUTSCHE BOERSE TradeGate (TG), and BOERSE_FRANKFURT_ZERTIFIKATE (SC).

I don't know.

US exchanges such as NYSE and Nasdaq operate with pre-market hours from 04:00-09:30, regular hours from 09:30-16:00, and post-market hours from 16:00-20:00, America/New_York time.

Fri Apr 18 20:00
</h1>

<h1 id="1b386594"><span style="font-size:18px;">
Excellent! Though, despite my best effort I could not convince Gemini to apply date correction (during chaining) based on holiday. It simply wasn't stable enough to be useful. I would either have to add a holiday data set, or (what I chose) apply a quick temporary fix. A real-time API endpoint may fail due to a holiday being selected as the date. If that happens I'll just retry Thursday if the failure happened on Friday, likewise choosing Friday if the failure happened on Monday. Crude but simple for foundational purposes.
</span></h1>

<h1 id="122c9ccb">Declaring the Function Calling Metadata</h1>

<p><span style="font-size:18px;">
Our Function Calling expert will chain together the other experts we've implemented thus far. It also provides the final response through augmentation. This time using the tools as a source of grounding truth. It'd like to say it's all truth organised by topic and other metadata. It's still a precarious situation if Essy incidently chains into mining data on another topic. We want Amazon to be the owner of MGM Studio's not MGM Resorts International. We also don't want a summary to include another company unless that company is a peer.
</span></p>

<h1 id="a7798460"><span style="font-size:18px;">
The function calling metadata is thus extremely important. It needs to combine our other experts with the real-time api's data. Essy will use two API providers as sources of finance data. The primary motivation being tha
</span></h1>

<h1 id="a7798460">t each provider has limits in their own way, yet both are useful in their own own way. This is useful anywhere you need a broad spectrum of sources of truth. At metadata creation I'll adopt the naming convention of appending the provider (if any) id. This helps keep functions more understandable when you know which provider you're dealing with.</h1>

<pre><code id="0669d563">
# Declare the callable functions using OpenAPI schema
get_symbol_1 = types.FunctionDeclaration(
    name="get_symbol_1",
    description="""Search for the stock ticker symbol of a given company, security, isin or cusip. Each ticker
                   entry provides a description, symbol, and asset type. If this doesn't help you should try
                   calling get_wiki_tool_response next.""",
    parameters={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": """The company, security, isin or cusip to search for a symbol."""
            },
            "exchange": {
                "type": "string",
                "description": """The exchange code used to filter results. When not specified the default exchange
                                  code you should use is 'US' for the US exchanges. A dictionary mapping all supported
                                  exchange codes to their names be retrieved by calling get_exchange_codes_1.
                                  Search for an exchange code to use by calling get_exchange_code_1, specifying the
                                  exchange code to search for."""
            },
</code></pre>

<pre><code id="0669d563">
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            }
        },
        "required": ["q", "exchange", "query"]
    }
)

get_name_1 = types.FunctionDeclaration(
    name="get_name_1",
    description="""Search for the name associated with a stock ticker or symbol's company, security, isin or cusip.
    Each ticker entry provides a description, matching symbol, and asset type.""",
    parameters={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": """The symbol or ticker to search for."""
            },
            "exchange": {
                "type": "string",
                "description": """The exchange code used to filter results. When not specified the default exchange
                                  code you should use is 'US' for the US exchanges. A dictionary mapping all supported
                                  exchange codes to their names be retrieved by calling get_exchange_codes_1.
                                  Search for an exchange code to use by calling get_exchange_code_1, specifying the
                                  exchange code to search for."""
            },
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            },
            "company": {
                "type": "string",
                "description": "The company you're searching for."
            }
        },
        "required": ["q", "exchange", "query", "company"]
    }
)

get_symbol_quote_1 = types.FunctionDeclaration(
    name="get_symbol_quote_1",
    description="""Search for the current price or quote of a stock ticker or symbol. The response is
                   provided in json format. Each response contains the following key-value pairs:

                   c: Current price,
                   d: Change,
                  dp: Percent change,
                   h: High price of the day,
                   l: Low price of the day,
                   o: Open price of the day,
                  pc: Previous close price,
                   t: Epoch timestamp of price in seconds.

                   Parse the response and respond according to this information.""",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker symbol for a company, security, isin, or cusip."
            },
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            },
            "exchange": {
                "type": "string",
                "description": "The exchange code used to filter quotes. This must always be 'US'."
            }
        },
        "required": ["symbol", "query", "exchange"]
    }
)

get_local_datetime_1 = types.FunctionDeclaration(
    name="get_local_datetime_1",
    description="""Converts an array of timestamps from epoch time to the local timezone format. The result is an array
                   of date and time in locale appropriate format. Suitable for use in a locale appropriate response.
                   Treat this function as a vector function. Always prefer to batch timestamps for conversion. Use this
                   function to format your final response.""",
    parameters={
        "type": "object",
        "properties": {
            "t": {
                "type": "array",
                "description": """An array of timestamps in seconds since epoch to be converted. The order of
                                  timestamps matches the order of conversion.""",
                "items": {
                    "type": "integer"
                }
            }
        },
</code></pre>

<pre><code id="0669d563">
        "required": ["t"]
    }
)

get_market_status_1 = types.FunctionDeclaration(
    name="get_market_status_1",
    description="""Get the current trading status of specified exchanges. Possible values are:
                   Market is open, Market is closed, Pre-market hours, Post-market hours.""",
    parameters={
        "type": "object",
        "properties": {
            "exchange": {
                "type": "string",
                "description": """The exchange code to check the status for. A dictionary mapping all supported
                                  exchange codes to their names be retrieved by calling get_exchange_codes_1.
                                  Search for an exchange code to use by calling get_exchange_code_1, specifying the
                                  exchange code to search for."""
            }
        },
        "required": ["exchange"]
    }
)

get_company_peers_1 = types.FunctionDeclaration(
    name="get_company_peers_1",
    description="""Get a list of company peers. The response is provided in json format. Each response contains
                   the following key-value pairs:

                   symbol: The query com
</code></pre>

<pre><code id="0669d563">
pany's stock ticker symbol,
                   peers: A list containing the peers.

                   Each peers entry contains the following key-value pairs:

                   symbol: The peer company's stock ticker symbol,
                   name: The peer company's name.

                   Parse the response and respond according to this information.""",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "The stock ticker symbol of a company to obtain peers."
            },
            "grouping": {
                "type": "string",
                "description": """Specify the grouping category for choosing peers. When not specified the default
                                  category is subIndustry. This parameter may be one of the following values:
                                  sector, industry, subIndustry."""
            },
            "exchange": {
                "type": "string",
                "description": """The exchange code used to filter results. When not specified the default exchange
                                  code you should use is 'US' for the US exchanges. A dictionary mapping all supported
                                  exchange codes to their names be retrieved by calling get_exchange_codes_1.
                                  Search for an exchange code to use by calling get_exchange_code_1, specifying the
                                  exchange code to search for."""
            },
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            }
        },
        "required": ["symbol", "grouping", "exchange", "query"]
    }
)

get_exchange_codes_1 = types.FunctionDeclaration(
    name="get_exchange_codes_1",
    description="""Get a dictionary mapping all supported exchange codes to t
</code></pre>

<pre><code id="0669d563">
heir names."""
)

get_exchange_code_1 = types.FunctionDeclaration(
    name="get_exchange_code_1",
    description="""Search for the exchange code to use when filtering by exchange. The result will be one or
                   more exchange codes provided as a comma-separated string value.""",
    parameters={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": "Specifies which exchange code to search for."
            }
        },
        "required": ["q"]
    }
)

get_financials_1 = types.FunctionDeclaration(
    name="get_financials_1",
    description="""Get company basic financials such as margin, P/E ratio, 52-week high/low, etc. Parse the response for
                   key-value pairs in json format and interpret their meaning as stock market financial indicators.""",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol for a company."
            },
            "metric": {
                "type": "string",
                "description": "It must always be declared as the value 'all'"
            },
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            }
        },
        "required": ["symbol", "metric", "query"]
    }
)

get_company_news_1 = types.FunctionDeclaration(
    name="get_company_news_1",
    description="Retrieve the most recent news articles related to a specified ticker.",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol for a company.",
            },
            "from": {
                "type": "string",
                "format": "date-time",
</code></pre>

<pre><code id="0669d563">
                "description": """Return news from this date.
                                  Accepted format: YYYY-MM-DD."""
            },
            "to": {
                "type": "string",
                "format": "date-time",
                "description": """Return news up to this date.
                                  Accepted format: YYYY-MM-DD."""
            }
        },
        "required": ["symbol", "from", "to"]
    }
)

get_daily_candlestick_2 = types.FunctionDeclaration(
    name="get_daily_candlestick_2",
    description="""Get daily OHLCV data for a single ticker symbol on a specific date.""",
    parameters={
        "type": "object",
        "properties": {
            "stocksTicker": {
                "type": "string",
                "description": "The ticker symbol of the stock/equity."
            },
            "date": {
                "type": "string",
                "format": "date",
                "description": """The date for which to retrieve data.
                                  Accepted format: YYYY-MM-DD."""
            },
            "adjusted": {
                "type": "string",
                "description": """Whether or not the results are adjusted for splits.
                                  By default, results are adjusted. Set this to false to get results that are NOT
                                  adjusted for splits."""
            }
        },
        "required": ["stocksTicker", "date", "adjusted"]
    }
)

get_custom_candlestick_2 = types.FunctionDeclaration(
    name="get_custom_candlestick_2",
    description="""Get aggregate OHLCV data for a given ticker symbol and date range.""",
    parameters={
        "type": "object",
        "properties": {
            "stocksTicker": {
                "type": "string",
                "description": "The ticker symbol of the stock/equity."
            },
            "multiplier": {
                "type": "integer",
                "description": """The size of the timespan multiplier. Must be an integer greater than 0."""
            },
            "timespan": {
                "type": "string",
                "description": """The size of the time window.
                                  Valid values are: minute, hour, day, week, month, quarter, year."""
            },
            "from": {
                "type": "string",
                "format": "date",
                "description": """The start date of the aggregate time window.
                                  Accepted format: YYYY-MM-DD."""
            },
            "to": {
                "type": "string",
                "format": "date",
                "description": """The end date of the aggregate time window.
                                  Accepted format: YYYY-MM-DD. The de
</code></pre>

<pre><code id="0669d563">
fault value is now's date."""
            },
            "adjusted": {
                "type": "string",
                "description": """May be true or false. Indicated whether or not the results are adjusted for splits.
                                  By default, results are adjusted. Set this to false to get results that are NOT
                                  adjusted for splits."""
            },
            "sort": {
                "type": "string",
                "description": """May be one of asc or desc. asc will sort by timestmap in ascending order. desc will
                                  sort by timestamp in descending order."""
            },
            "limit": {
                "type": "integer",
                "description": """Set the number of base aggregates used to create this custom result. The default is
                                  5000 and the maximum is 50000."""
            },
            "query": {
                "type": "string",
                "description": "The question you're attempting to answer."
            }
        },
        "required": ["stocksTicker", "multiplier", "timespan", "from", "to", "query", "adjusted", "sort", "limit"]
    },
)

get_last_market_close = types.FunctionDeclaration(
    name="get_last_market_close",
    description="""Get the date and time of the US exchange market's last close. Provides the last US market close in
                   locale appropriate format."""
)

get_ticker_overview_2 = types.FunctionDeclaration(
    name="get_ticker_overview_2",
    description="""Retrieve comprehensive details for a single ticker symbol. It's a deep look into a company’s
    fundamental attributes, including its primary exchange, standardized identifiers (CIK, composite FIGI,
    share class FIGI), market capitalization, industry classification, and key dates. Also includes branding assets in
    the form of icons and logos.
    """,
    parameters={
        "type": "object",
</code></pre>

<pre><code id="0669d563">
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol of a company."
            }
        },
        "required": ["ticker"]
    }
)

get_recommendation_trends_1 = types.FunctionDeclaration(
    name="get_recommendation_trends_1",
    description="""Get the latest analyst recommendation trends for a company.
                The data includes the latest recommendations as well as historical
                recommendation data for each month. The data is classified according
                to these categories: strongBuy, buy, hold, sell, and strongSell.
                The date of a recommendation indicated by the value of 'period'.""",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol for a company."
            }
        },
        "required": ["symbol"]
    }
)

get_news_with_sentiment_2 = types.FunctionDeclaration(
    name="get_news_with_sentiment_2",
    description="""Retrieve the most recent news articles related to a specified ticker. Each article includes
                   comprehensive coverage. Including a summary, publisher information, article metadata,
                   and sentiment analysis.""",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol for a company."
            },
            "published_utc": {
                "type": "string",
                "format": "date-time",
                "description": """Return results published on, before, or after this date in UTC. An example date
                                  looks like this 2025-04-10T00:24:00Z. The default is to omit this value unless
                                  specified."""
            },
            "order": {
</code></pre>

<pre><code id="0669d563">
                "type": "string",
                "description": """Must be asc if ascending order, or desc for decending ordering.
                                  When order is omitted default to ascending ordering.
                                  Ordering will be based on the parameter: sort."""
            },
            "limit": {
                "type": "integer",
                "description": """This is allowed to range from 100 to 1000."""
            },
            "sort": {
                "type": "string",
                "description": """The sort field used for ordering. This value must
                                  always be published_utc."""
            }
        },
        "required": ["ticker", "order", "limit", "sort"]
    }
)

get_rag_tool_response = types.FunctionDeclaration(
    name="get_rag_tool_response",
    description="""A database containing useful financial information. Always check here for answers first.""",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "A question needing an answer. Asked as a simple string."
            }
        }
    }
)

get_wiki_tool_response = types.FunctionDeclaration(
    name="get_wiki_tool_response",
    description="""Answers questions that still have unknown answers. Retrieve a wiki page related to a company,
                   product, or service. Each web page includes detailed company information, financial indicators,
                   tickers, symbols, history, and products and services.""",
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The question's company or product. Just the name and no other details."
            },
            "q": {
                "type": "string",
                "description": "The complete, unaltered, que
</code></pre>

<pre><code id="0669d563">
ry string."""
            }
        },
        "required": ["id", "q"]
    }
)

get_search_tool_response = types.FunctionDeclaration(
    name="get_search_tool_response",
    description="Answers questions that still have unknown answers. Use it after checking all your other tools.",
    parameters={
        "type": "object",
        "properties": {
            "q": {
                "type": "string",
                "description": "The question needing an answer. Asked as a simple string."
            },
            "id": {
                "type": "string",
                "description": "The question's company or product. In one word. Just the name and no other details."
            }
        },
        "required": ["q", "id"]
    }
)
</code></pre>

<h1 id="a900edac">Implementing the Function Calls</h1>

<p><span style="font-size:18px;">
One downside of this part being the main part was the lack of time to refactor this part more. Our formative Essy implements as much useful data from two finacial APIs. In order to use it you will need to declare secrets for <a class="anchor-link" href="https://finnhub.io/dashboard">Finnhub</a> and <a class="anchor-link" href="https://polygon.io/dashboard">Polygon</a> finance APIs. Register at their respective sites for your free API key. Then import the secret using the same method as how you setup Google's API key.
</span></p>

<pre><code id="6d747efe">
# Implement the callable functions and the function handler

def ask_rag_tool(content):
    return tool_rag.generate_answer(content["question"], max_sources = 20).text

def ask_wiki_tool(content):
    return tool_wiki.generate_answer(content["q"], content["id"])

def ask_search_tool(content):
    return tool_ground.generate_answer(content["q"], content["id"])

def rag_exchange_codes_1(content):
    response = tool_rag.get_exchanges_csv("""Give me a dictionary in string form. It must contaihttps://api.polygon.io/v3/reference/tickers/AAPL?apiKey=4xJe226Z23RZmEc1bN8az1zz4pmNWdOpn key:value pairs
                                             mapping exchange code to name. Just the dictionary string.
                                             Omit all other information or details. Do not chat or use sentences.""")
    codes = list(ast.literal_eval(response.text.strip("`")).items())
    return codes

def rag_exchange_code_1(content):
    codes = tool_rag.get_exchanges_csv(f"""What is the {content} exchange code? Return only the exchange codes
                                           as a list in string form. Just the list string.
                                           Omit all other information or details. Do not chat or use sentences.""").text
    return ast.literal_eval(codes)

def rag_last_market_close(content):
    return dateutil.parse(tool_rag.get_exchanges_csv(f"""Answer based on your knowledge of exchange operating hours.
    The exchanges are all closed on weekends. Consider after-market hours as the market being open. When did the US
    exchanges last close? Provide the date and time in Eastern Time. The day should be one of: Mon, Tue, Wed, Thu, Fri.

    The current date and time is: {datetime.now().strftime('%c')}

    Provide only the date and time. Omit all other information or details. Do not chat or use sentences.""").text).strftime('%c')

def get_similarity_score(content):
    similarity = client.models.embed_content(
        model="models/text-embedding-004",
        contents=content,
        con
</code></pre>

<pre><code id="6d747efe">
fig=types.EmbedContentConfig(task_type="semantic_similarity"))
    df = pandas.DataFrame([e.values for e in similarity.embeddings], index=content)
    score = df @ df.T
    return score.iloc.iloc

def impl_get_symbol_1(content, by_name: bool = True):
    response = tool_rag.get_api_documents(content["query"], content["q"], "get_symbol_1")
    if len(response) == 0: # index  for document content
        url = f"https://finnhub.io/api/v1/search?q={content['q']}&exchange={content['exchange']}&token={FINNHUB_API_KEY}"
        try:
            response = json.loads(requests.get(url).text)
        except:
            return "I don't know."
        else:
            matches = []
            max_failed_match = len(response["result"]) if not by_name else 3
            p_desc_match = 0.80
            p_symb_match = 0.95
            if response["count"] > 0:
                for result in tqdm(response["result"], desc="Score similarity to query"):
                    if max_failed_match > 0:
                        desc = [content['q'].upper(), result["description"].split("-", -1)]
                        symb = [content['q'].upper(), result["symbol"]]
                        if by_name and get_similarity_score(desc) > p_desc_match:
                            matches.append(result["symbol"])
                        elif not by_name and get_similarity_score(symb) > p_symb_match:
                            matches.append(result["description"])
                            max_failed_match = 0
                        else:
                            max_failed_match -= 1
            if len(matches) > 0:
                tool_rag.add_api_document(content["query"], ", ".join(matches), content["q"], "get_symbol_1")
                return ", ".join(matches)
            else:
                return "I don't know."
    else:
        doc = ast.literal_eval(response)
        return doc["answer"]

def impl_get_name_1(content):
    return impl_get_symbol_1(content, by_name = False)

def impl_get_qu
</code></pre>

<pre><code id="6d747efe">
ote_1(content):
    quotes = tool_rag.get_api_documents(content["query"], content["symbol"], "get_quote_1")
    isOpen = dict(impl_get_market_status_1(content))["isOpen"]
    if len(quotes) == 0 or isOpen:
        return get_current_price_1(content)
    else:
        last_close = rag_last_market_close(content).timestamp()
        for quote in quotes: # index for metadata
            if last_close == quote["timestamp"]:
                return quotes
        return get_current_price_1(content)

def get_current_price_1(content):
    url = f"https://finnhub.io/api/v1/quote?symbol={content['symbol']}&token={FINNHUB_API_KEY}"
    # This is a high-demand endpoint. Expect random failure under heavy (free) use.
    try:
        response = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if len(response) > 0 and response["t"] > 0:
            tool_rag.add_quote_document(content["query"], response, content["symbol"], response["t"], "get_quote_1")
            return list(response.items())
        return "I don't know."

def impl_get_market_status_1(content):
    url = f"https://finnhub.io/api/v1/stock/market-status?exchange={content['exchange']}&token={FINNHUB_API_KEY}"
    try:
        response = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if len(response) > 0:
            return list(response.items())
        return "I don't know."

def impl_get_peers_1(content):
    docs = tool_rag.get_peers_document(content["query"], content["symbol"], content['grouping'])
    if len(docs) == 0: # index  for document content
        url = f"https://finnhub.io/api/v1/stock/peers?symbol={content['symbol']}&grouping={content['grouping']}&token={FINNHUB_API_KEY}"
        try:
            peers = json.loads(requests.get(url).text)
        except:
            return "I don't know."
        else:
            if len(peers) > 0:
                names = []
                for peer in peers:
</code></pre>

<pre><code id="6d747efe">
                    if peer == content["symbol"]:
                        continue # skip including the query symbol in peers (included in metadata anyway)
                    name_lookup = dict(q=peer, exchange=content["exchange"], query=content["query"])
                    name = impl_get_name_1(name_lookup)
                    if name != "I don't know.":
                        p = {"symbol": peer, "name": name}
                        names.append(p)
                peers = {"symbol": content["symbol"], "peers": names}
                tool_rag.add_peers_document(content["query"], peers, content["symbol"], "get_peers_1", content['grouping'])
                return list(peers.items())
            return "I don't know."
    else:
        peers = ast.literal_eval(docs)["answer"] # The first document should be most relevant.
        return list(peers.items())

def impl_local_datetime_1(content):
    local_t = []
    for timestamp in content["t"]:
        local_t.append(datetime.fromtimestamp(timestamp).strftime('%c'))
    return local_t

def impl_get_financials_1(content):
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={content['symbol']}&metric={content['metric']}&token={FINNHUB_API_KEY}"
    try:
        fin = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if not fin:
            return "I don't know."
        return list(fin.items())

def impl_get_news_1(content):
    url = f"https://finnhub.io/api/v1/company-news?symbol={content['symbol']}&from={content['from']}&to={content['to']}&token={FINNHUB_API_KEY}"
    try:
        news = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if len(news) == 0:
            return "I don't know."
        return news

def impl_daily_candle_2(content):
    url = f"https://api.polygon.io/v1/open-close/{content['stocksTicker']}/{content['date']}?adjusted={content['adjusted']}&apiKey={POLYGON_API_KEY}"
    try:
        daily_candle = ast.literal_eval(req
</code></pre>

<pre><code id="6d747efe">
uests.get(url).text)
    except:
        return "I don't know."
    else:
        if daily_candle["status"] == "OK":
            return list(daily_candle.items())
        else:
            date = dateutil.parse(content["date"])
            new_date = None
            if date.weekday() == 4: # index 4 for friday
                new_date = date - timedelta(days=1)
            elif date.weekday() == 0: # index 0 for monday
                new_date = date - timedelta(days=3)
            if new_date is None:
                return "I don't know."
            content["date"] = new_date.strftime("%Y-%m-%d")
            return impl_daily_candle_2(content)

def impl_custom_candle_2(content):
    url = f"""https://api.polygon.io/v2/aggs/ticker/{content['stocksTicker']}/range/{content['multiplier']}/{content['timespan']}/{content['from']}/{content['to']}?adjusted={content['adjusted']}&sort={content['sort']}&limit={content['limit']}&apiKey={POLYGON_API_KEY}"""
    try:
        custom_candle = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if custom_candle["status"] == "OK":
            return list(custom_candle.items())
        return "I don't know."

def impl_ticker_overview_2(content):
    url = f"https://api.polygon.io/v3/reference/tickers/{content['ticker']}?apiKey={POLYGON_API_KEY}"
    try:
        overview = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if overview["status"] == "OK":
            return list(overview.items())
        return "I don't know."

def impl_trends_1(content):
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={content['symbol']}&token={FINNHUB_API_KEY}"
    try:
        trends = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if len(trends) > 0:
            return trends
        return "I don't know."

def impl_get_news_2(content):
    url = f"https://api.polygon.io/v2/reference/news?
</code></pre>

<pre><code id="6d747efe">
ticker={content['ticker']}&order={content['order']}&limit={content['limit']}&sort={content['sort']}&apiKey={POLYGON_API_KEY}"
    try:
        news = json.loads(requests.get(url).text)
    except:
        return "I don't know."
    else:
        if news["status"] == "OK":
            return list(news.items())
        return "I don't know."

finance_tool = types.Tool(
    function_declarations=[
        get_symbol_1,
        get_name_1,
        get_symbol_quote_1,
        get_market_status_1,
        get_company_peers_1,
        get_local_datetime_1,
        get_last_market_close,
        get_exchange_codes_1,
        get_exchange_code_1,
        get_financials_1,
        get_company_news_1,
        get_daily_candlestick_2,
        get_custom_candlestick_2,
        get_ticker_overview_2,
        get_recommendation_trends_1,
        get_news_with_sentiment_2,
        get_rag_tool_response,
        get_wiki_tool_response,
        get_search_tool_response
    ]
)

function_handler = {
    "get_symbol_1": impl_get_symbol_1,
    "get_name_1": impl_get_name_1,
    "get_symbol_quote_1": impl_get_quote_1,
    "get_market_status_1": impl_get_market_status_1,
    "get_company_peers_1": impl_get_peers_1,
    "get_local_datetime_1": impl_local_datetime_1,
    "get_last_market_close": rag_last_market_close,
    "get_exchange_codes_1": rag_exchange_codes_1,
    "get_exchange_code_1": rag_exchange_code_1,
    "get_financials_1": impl_get_financials_1,
    "get_company_news_1": impl_get_news_1,
    "get_daily_candlestick_2": impl_daily_candle_2,
    "get_custom_candlestick_2": impl_custom_candle_2,
    "get_ticker_overview_2": impl_ticker_overview_2,
    "get_recommendation_trends_1": impl_trends_1,
    "get_news_with_sentiment_2": impl_get_news_2,
    "get_rag_tool_response": ask_rag_tool,
    "get_wiki_tool_response": ask_wiki_tool,
    "get_search_tool_response": ask_search_tool
}
</code></pre>

<pre><code id="0727fbb8">
# Implement the function calling expert.

def send_message(prompt):
    #display(Markdown("#### Prompt"))
    #print(prompt, "\n")
    # Define the user prompt part.
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    contents += """
    Give a concise, and detailed summary. Use information that you learn from the API responses.
    Use your tools and function calls according to the rules. Convert any all-upper case identifiers
    to proper case in your response. Convert any abbreviated or shortened identifiers to their full forms.
    """
    # Enable system prompt, function calling and minimum-randomness.
    config_fncall = types.GenerateContentConfig(
        system_instruction=instruction,
        tools=[finance_tool],
        temperature=0.0
    )
    # Handle cases with multiple chained function calls.
    function_calling_in_process = True
    while function_calling_in_process:
        # Send the user prompt and function declarations.
        response = client.models.generate_content(
            model=project_model, config=config_fncall, contents=contents
        )
        # A part can be a function call or natural language response.
        for part in response.candidates.content.parts:
            if function_call := part.function_call:
                # Extract the function call.
                fn_name = function_call.name
                #display(Markdown("#### Predicted function name"))
                #print(fn_name, "\n")
                # Extract the function call arguments.
                fn_args = {key: value for key, value in function_call.args.items()}
                #display(Markdown("#### Predicted function arguments"))
                #print(fn_args, "\n")
                # Call the predicted function.
                api_response = function_handler[fn_name](fn_args)[:20000] # Stay within the
</code></pre>

<pre><code id="0727fbb8">
input token limit
                #display(Markdown("#### API response"))
                #print(api_response[:500], "...", "\n")
                # Create an API response part.
                api_response_part = types.Part.from_function_response(
                    name=fn_name,
                    response={"content": api_response},
                )
                # Append the model's function call part.
                contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
                # Append the api response part.
                contents.append(types.Content(role="user", parts=[api_response_part]))
            else:
                # The model gave a natural language response
                function_calling_in_process = False
                break # No more parts in response.
        if not function_calling_in_process:
            break # The function calling chain is complete.

    # Show the final natural language summary
    display(Markdown("#### Natural language response"))
    display(Markdown(response.text.replace("$", "\\\\$\")))
</code></pre>

<h1 id="6b57d063">Ask a question</h1>

<pre><code id="524c83a6">
send_message(
    '''Tell me about Google's current share price,
    short-term trends, and bullish versus bearish predictions.
    Include sentiment analysis please.''')
</code></pre>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="output_subarea output_stream output_stderr output_text">
<pre>Score similarity to query: 100%|██████████| 11/11 [00:00&lt;00:00, 19.53it/s]
Score wiki search by similarity to topic: 0it [00:00, ?it/s]
Generate wiki embedding: 0it [00:00, ?it/s]
Generate grounding embedding: 0it [00:00, ?it/s]
</pre>
</div>
</div>
</div>
<div class="output_area">
<div class="output_subarea output_display">
<h4 id="524c83a6">Natural language response</h4>
</div>
</div>
<div class="output_area">
<div class="output_subarea output_markdown rendered_html">
Okay, here's a summary of Google's (Alphabet Inc.) stock information:

**Current Stock Price:** As of April 17, 2025, the price of Google (GOOG) is \$\$153.36. The price has decreased by -1.38% in the past 24 hours.

**Short-Term Trends:** The stock has fallen by -3.40% compared to the previous week and -7.59% over the last month. Over the last year, the stock has shown a -2.44% decrease. A short-term trend indicates the stock is expected to fall -24.47% during the next 3 months.

**Bullish vs. Bearish Predictions:**

*   **Bullish:** Analysts' opinions suggest a maximum estimate of \$\$250.00 and a minimum estimate of \$\$159.00. Morgan Stanley predicts that the stock could climb above \$\$200. CoinCodex forecasts the stock to rise in value, with the most significant gains expected in the second half of the year. A doji candlestick pattern has formed, which is considered a bullish pattern.
*   **Bearish:** A leading financial portal, Traders Union, has painted a bearish outlook, providing a negative price prediction for April 2025. The Google stock holds sell signals from both short and long-term Moving Averages, giving a more negative forecast for the stock. Given the current short-term trend, the stock is expected to fall -24.47% during the next 3 months.

**Sentiment Analysis:** The overall sentiment for investing in Google is positive, with a sentiment score of 70 out of 100, according to data from top investing forums. However, this is down compared to the 30-day moving average. Google overperforms a majority of its industry peers when it comes to sentiment, ranking in the 75th percentile.

**Additional Points:** Alphabet is scheduled to release its next earnings report on April 24, 2025. Analysts anticipate that first-quarter revenue rose by 10.8% to \$\$89.3 billion. The average price target from analysts is \$\$200.74. The stock faces an important hurdle near the \$\$160 mark, where the 200-day simple moving average currently sits on the daily chart.
</div>
</div>
</div>
</div>

<h1 id="d317c3c6">Conclusion</h1>

<p><span style="font-size:18px;">
For now that will have to do. Our Essy has a solid foundation but more could be done to organise metadata. No evaluation or validation has been performed (except fuzzing the prompt). Next steps include restructuring the vector database based on lessons learned. That'll be followed by plotting, multi-modal, and structured output. The last close date (generative) function can be temperamental. In the same way Gemini always feels regarding dates. I've learnt so much. I'm happy I decided to participate in the event! It really has been a joy to see Essy grow from random chat with Gemini into the foundation for a good-broker buddy. I hope you enjoy playing with this edition as much as I enjoyed building it!
</span></p>

</body>
</html>
```
