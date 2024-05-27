from openai import OpenAI
import os
import pandas as pd
import re

def generate_diagnosis_prompt(row):
    """
    Using the | separated string of symptoms generate the prompt to elicit a list of 10 possible diagnoses.
    Make sure each symptom is separated by a comma and a space.
    """
    symptomps = row["Short terms"]
    title = row["Title"]
    symptomps = symptomps.replace("|", ", ")
    prompt = f"You are an expert Harvard Medical School trained physician who is excellent at making accurate diagnoses. {title}. They have the following symptoms: {symptomps}. Give me the 10 most likely diagnoses in order of decreasing probability: \n1."
    return prompt

def generate_chat_completions_messages(row):
    """
    Using the | separated string of symptoms generate the prompt to elicit a list of 10 possible diagnoses.
    Make sure each symptom is separated by a comma and a space.
    """
    system_message = f"You are an expert Harvard Medical School trained physician who is excellent at making accurate diagnoses."
    symptomps = row["Short terms"]
    title = row["Title"]
    symptomps = symptomps.replace("|", ", ")
    user_message = f"{title}. They have the following symptoms: {symptomps}. Give me the 10 most likely diagnoses in order of decreasing probability: \n1."
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
    ]
    return messages

def parse_diagnosis_list_response(response):
    """
    Parse the response from the openai api call
    """
    diagnoses = response.choices[0].message.content.strip().split("\n")
    diagnoses = [diagnosis.strip() for diagnosis in diagnoses]
    return " ".join(diagnoses)

def parse_answer_check_response(response):
    """
    Parse the response from the openai api call whith format "Rank: 3" if the diagnosis was the third item in the list of top 10 and "Rank: -1" if it wasn't any of the top 10.
    """
    match = re.search("Rank: [0-9]+", response.choices[0].message.content.strip())
    if match:
        return int(match.group().split("Rank: ")[1])
    else:
        return -1


def get_diagnosis_rank(actual_diagnosis, diagnosis_list_str):
    openai = OpenAI(api_key=OPENAI_API_KEY)
    check_answer_prompt = "\nThe actual diagnosis is {}. Was this diagnosis in your top 10? If so return it's rank".format(actual_diagnosis)
    check_answer_response = openai.chat.completions.create(
        model=OPENAI_CHAT_COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": "Given the following diagnosis list, rank where the actual diagnosis is in the list. E.g. return 'Rank: 3' if it's rank was 3 and 'Rank: -1' if it's not in the top 10 above.\n"},
            {"role": "assistant", "content": diagnosis_list_str},
            {"role": "user", "content": check_answer_prompt}]
        )
    corect_answer_index = parse_answer_check_response(check_answer_response)
    return corect_answer_index
    
def add_gpt_predictions(row, auto_check_answer=True):
    openai = OpenAI(api_key=OPENAI_API_KEY)
    response = openai.chat.completions.create(
        model=OPENAI_CHAT_COMPLETIONS_MODEL,
        messages = generate_chat_completions_messages(row)
    )
    diagnoses = parse_diagnosis_list_response(response)
    if auto_check_answer:
        response_message = response.choices[0].message.content
        rank = get_diagnosis_rank(row['ActualDiagnosis'], response_message)
    else:
        rank = None
    return diagnoses, rank


def main(df, auto_check_answer=True):
    all_gpt_diagnoses, gpt_correct_answer_indices = [], []
    new_rows = []
    for i, row in df.iterrows():
        gpt_diagnoses, gpt_rank = add_gpt_predictions(row, auto_check_answer=auto_check_answer)
        all_gpt_diagnoses.append(gpt_diagnoses)
        gpt_correct_answer_indices.append(gpt_rank)
        row["GPTDiagnoses"] = gpt_diagnoses
        if auto_check_answer:
            row["GPTCorrectDiagnosisIndex"] = gpt_rank
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    return new_df

def top_k_accuracy(df, k, column_name="GPTCorrectDiagnosisIndex"):
    gpt_accuracy = ((df[column_name] > 0) & (df[column_name] < k+1)).sum() / len(df)
    return gpt_accuracy

if __name__ == "__main__":

    OPENAI_CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo"
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    print("OPENAI_API_KEY", OPENAI_API_KEY)
    
    # GENERATE GPT DIAGNOSES
    df = pd.read_csv('data/cases_df.csv')
    df = df.iloc[:3, :]
    results_df = main(df, auto_check_answer=True)
    print(results_df.loc[:, ["ActualDiagnosis", "GPTDiagnoses", "GPTCorrectDiagnosisIndex"]])
    results_df.to_csv("data/{}_results_df.csv".format(OPENAI_CHAT_COMPLETIONS_MODEL), index=False, header=True)

    # EVALUATE GPT ACCURACY
    results_df = pd.read_csv("data/{}_results_df.csv".format(OPENAI_CHAT_COMPLETIONS_MODEL))
    print("Total number of cases: {}".format(len(results_df)))
    for k in [1, 5, 10]:
        gpt_accuracy = top_k_accuracy(results_df, k)
        print("GPT top {} accuracy: {:.2f}".format(k, gpt_accuracy))
        print("---------------------------------")
