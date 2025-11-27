import os
import nltk
nltk.download('punkt_tab')

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, AbstractiveSummaryAction
from dotenv import load_dotenv

# For sentence ranking
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

load_dotenv()

# Azure credentials
endpoint = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
key = os.environ.get("AZURE_LANGUAGE_KEY")

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Meeting note
document = [
    "On 27 Nov 2025, the team reviewed Project Alpha timelines, confirming that Phase 1 deliverables are due by 10 Dec. "
    "Budget allocation for marketing was discussed, with approval still pending. "
    "The team agreed to adopt a new task management tool starting next week. "
    "Client feedback on the prototype was received, indicating minor design adjustments are required. "
    "Risks were identified, including potential supplier delays and resource constraints. "
    "Action items were assigned, with Ravi handling design revisions and Priya managing the testing schedule. "
    "The next meeting is scheduled for 4 Dec 2025 at 10:00 AM."
]

if __name__ == "__main__":
    # Azure abstractive summarization
    actions = [AbstractiveSummaryAction(max_sentence_count=10, order_by="Rank")]
    poller = client.begin_analyze_actions(document, actions=actions)
    response = poller.result()

    for action_results in response:
        for result in action_results:
            if result.kind == "AbstractiveSummarization":
                full_summary = result.summaries[0].text
                print("=== Full Abstractive Summary ===\n")
                print(full_summary, "\n")

                # Generate executive summary (3 sentences) using sumy LexRank
                parser = PlaintextParser.from_string(full_summary, Tokenizer("english"))
                summarizer = LexRankSummarizer()
                top_sentences = summarizer(parser.document, sentences_count=3)

                
                top3 = top_sentences[:3]
                print("=== Executive 3-Sentence Summary ===\n")
                for sentence in top3:
                    print("•", str(sentence).strip())
                print()

            elif result.is_error:
                print(f"Error '{result.code}': {result.message}")
