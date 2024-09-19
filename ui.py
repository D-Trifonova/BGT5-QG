import streamlit as st

from utils import generate_answer, generate_question, generate_distractors, generate_test_question, \
    load_sentence_tokenizer
from pages import generate_single_answer_page, generate_single_question_page, generate_distractors_page, \
    generate_test_question_page, generate_tests

if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = load_sentence_tokenizer()

#st.title("Генериране на тестове")

pg = st.navigation([
    st.Page(generate_single_answer_page, title="Генериране на единичен отговор"),
    st.Page(generate_single_question_page, title="Генериране на единичен въпрос"),
    st.Page(generate_distractors_page, title="Генериране на дистрактори"),
    st.Page(generate_test_question_page, title="Генериране на единичен тестов въпрос"),
    st.Page(generate_tests, title="Генериране на тестови въпроси от файл"),
])
pg.run()
