import streamlit as st
from utils import generate_answer, generate_question, generate_distractors, generate_test_question
import random
import PyPDF2

MIN_SENTENCES_PER_CONTEXT = 3
MAX_SENTENCES_PER_CONTEXT = 7


def generate_single_answer_page():
    st.subheader("Въведете контекст за генериране на отговор")
    context = st.text_area("Контекст", height=300)

    if st.button("Генерирай отговор"):
        response = generate_answer(context)
        if "error_code" not in response:
            st.write("#### Генериран отговор:")
            st.write(response["answer"])
        else:
            st.error(f"Error {response['error_code']} generating answer: {response['error']}")


def generate_single_question_page():
    st.subheader("Въведете отговор и контекст за генериране на въпрос")
    context = st.text_area("Контекст", height=300)
    answer = st.text_input("Отговор")

    if st.button("Генерирай въпрос"):
        response = generate_question(answer, context)
        if "error_code" not in response:
            st.write("#### Генериран въпрос:")
            st.write(f"**Отговор:** {response['answer']}")
            st.write(f"**Въпрос:** {response['question']}")
        else:
            st.error(f"Error {response['error_code']} generating question: {response['error']}")

def generate_distractors_page():
    st.subheader("Въведете въпрос, отговор и контекст за генериране на дистрактори")
    context = st.text_area("Контекст", height=300)
    question = st.text_input("Въпрос")
    answer = st.text_input("Отговор")

    if st.button("Генерирай дистрактори"):
        response = generate_distractors(question, answer, context)
        if "error_code" not in response:
            st.write("#### Генерирани дистрактори:")
            for did, distractor in enumerate(response["distractors"], 1):
                st.write(f"**Дистрактор {did}:** {distractor}")
        else:
            st.error(f"Error {response['error_code']} generating distractors: {response['error']}")


def generate_test_question_page():
    st.subheader("Въведете контекст")
    context = st.text_area("Контекст", height=300)

    if st.button("Генерирай тестов въпрос"):
        response = generate_test_question(context)
        if "error_code" not in response:
            st.write("#### Генериран тестов въпрос:")
            st.write(f"**Въпрос:** {response['question']}")
            st.write(f"**Отговори:**")
            for aid, answer in enumerate(response["answers"]):
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;**{aid}.** {answer}")

            st.write(f"**Верен отговор:** {response['correct_answer_text']}")
            st.write(f"**ID на верен отговор:** {response['correct_answer_id']}")
        else:
            st.error(f"Error {response['error_code']} generating test question: {response['error']}")

def generate_tests():
    question_count = st.number_input("Брой въпроси:", min_value=1, max_value=10, step=1, format="%d",
                                     help="Въведете брой въпроси за генериране, като максимума е 10 въпроса наведнъж.")
    file = st.file_uploader("Изберете pdf файл", type=["pdf", "txt"])
    if file is not None:
        if file.type == "text/plain":
            content = file.read().decode()
        else:
            pdf_reader = PyPDF2.PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        sentences = st.session_state['tokenizer'].tokenize(content)

        text_questions = []
        error = False
        for _ in range(question_count):
            start_sentence = random.randrange(len(sentences))
            end_sentence = start_sentence + random.randint(MIN_SENTENCES_PER_CONTEXT, MAX_SENTENCES_PER_CONTEXT)
            selected_sentences = sentences[start_sentence:end_sentence]

            context = ""
            for sentence in selected_sentences:
                context += sentence.replace("\n", " ") + " "

            response = generate_test_question(context)
            if "error_code" not in response:
                response["context"] = context
                text_questions.append(response)
            else:
                st.error(f"Error {response['error_code']} generating test question: {response['error']}")
                error = True
                break
        if not error:
            st.write(text_questions)
