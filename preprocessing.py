import os
import json
import pandas as pd


def process_json_content(json_content):
    """
    Process JSON content to extract question-answer-label data.

    :param json_content: JSON data loaded into a Python dictionary.
    :return: DataFrame with question, answer, and label information.
    """
    rows = []
    conversations = json_content.get("dataset", {}).get("conversations", [])

    for conversation in conversations:
        utterances = conversation.get("utterances", [])

        for i in range(len(utterances) - 1):
            current_utterance = utterances[i]
            next_utterance = utterances[i + 1]

            if current_utterance["speaker_id"] != 0 and next_utterance["speaker_id"] == 0:
                question = current_utterance["utterance_text"]
                answer = next_utterance["utterance_text"]

                evaluations = next_utterance.get("utterance_evaluation", [])
                aggregated_labels = {}
                if evaluations:
                    aggregated_labels = {
                        key: all(evaluation.get(key, "no") == "yes" for evaluation in evaluations)
                        for key in evaluations[0].keys()
                    }

                rows.append({
                    "question": question,
                    "answer": answer,
                    "label": aggregated_labels
                })

    return pd.DataFrame(rows)


def process_all_json_files(directory_path):
    """
    Process all JSON files in the directory to extract question-answer-label data.

    :param directory_path: Path to the directory containing JSON files.
    :return: DataFrame with aggregated question-answer-label data from all files.
    """
    all_data = []
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(directory_path, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                json_content = json.load(file)
                df = process_json_content(json_content)
                if not df.empty:
                    all_data.append(df)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


directory_path = "dataset/Validation/02.라벨링데이터/VL_1.발화단위평가_여행_여가_취미"
output_csv_path = "test_data.csv"
result_df = process_all_json_files(directory_path)
result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')