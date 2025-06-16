import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", layout='wide')

# Sidebar navigation
st.sidebar.title("Graph Menu")
section = st.sidebar.radio("", [
    "Registrations Summary",
    "Age & Gender Insights",
    "SDG Security Questions",
    "Questionnaire Responses"
])

# Cached data loading and preprocessing
@st.cache_data
def load_data():
    household_df = pd.read_excel("Household_Details_14052025_to_16062025.xlsx")
    family_df = pd.read_excel("Household_Family_Members_14052025_to_16062025.xlsx")
    questionnaire_df = pd.read_excel("Questinnaire_Responses_14052025_to_16062025.xlsx")

    # Shared preprocessing
    household_df['created_at'] = pd.to_datetime(household_df['created_at'])
    household_df['date'] = household_df['created_at'].dt.date
    family_df['age'] = pd.to_numeric(family_df['age'], errors='coerce')

    return household_df, family_df, questionnaire_df

# Load data
household_df, family_df, questionnaire_df = load_data()

# Section: Registrations Summary
if section == "Registrations Summary":
    st.title("Volunteer Registrations Summary")
    total = household_df.groupby('volunteer_id').size().reset_index(name='total')
    avg = household_df.groupby(['volunteer_id', 'date']).size().groupby('volunteer_id').mean().reset_index(name='avg')
    df = pd.merge(total, avg, on='volunteer_id')
    df = df[~df['volunteer_id'].isin([3, 11])]
    df = df.sort_values(by='total', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df['total'], width=0.4, label='Total', color='skyblue')
    ax.bar([i + 0.2 for i in x], df['avg'], width=0.4, label='Avg/Day', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(df['volunteer_id'], rotation=45)
    ax.legend()
    ax.set_title("Total & Average Registrations per Volunteer")
    st.pyplot(fig)

# Section: Age & Gender Insights
elif section == "Age & Gender Insights":
    st.title("Age & Gender Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Gender Count")
        gender_total = family_df['gender'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(gender_total.index, gender_total.values, color=['#4A90E2', '#F78DA7', '#A0A0A0'][:len(gender_total)])
        for i, val in enumerate(gender_total.values):
            ax.text(i, val + 1, str(val), ha='center')
        st.pyplot(fig)

    with col2:
        st.subheader("Gender Count by Age Group")
        bins = [0, 2, 14, 30, 60, float('inf')]
        labels = ['0-2', '2-14', '14-30', '30-60', '60+']
        family_df['age_group'] = pd.cut(family_df['age'], bins=bins, labels=labels, right=False)
        age_gender = family_df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)

        fig, ax = plt.subplots()
        age_gender.plot(kind='bar', ax=ax, color=['#4A90E2', '#F78DA7', '#A0A0A0'][:len(age_gender.columns)])
        st.pyplot(fig)

# Section: SDG Security Questions
elif section == "SDG Security Questions":
    st.title("Security in Six Areas (Questions 91–96)")
    question_texts = {
        91: "How secure you feel in these six areas (food)?",
        92: "How secure you feel in these six areas (health)?",
        93: "How secure you feel in these six areas (environment)?",
        94: "How secure you feel in these six areas (financial)?",
        95: "How secure you feel in these six areas (shelter)?",
        96: "How secure you feel in these six areas (gender)"
    }
    sdg_colors = {
        91: '#E5243B', 92: '#4C9F38', 93: '#3F7E44',
        94: '#F36D25', 95: '#F89D2A', 96: '#C5192D'
    }

    q_six = questionnaire_df[questionnaire_df['question_id'].between(91, 96)]
    responses = q_six.groupby(['question_id', 'response']).size().unstack(fill_value=0)

    for start in [91, 94]:
        cols = st.columns(3)
        for i, qid in enumerate(range(start, start + 3)):
            with cols[i]:
                st.subheader(question_texts[qid])
                yes = responses.loc[qid].get('Yes', 0)
                no = responses.loc[qid].get('No', 0)
                fig, ax = plt.subplots()
                ax.bar(['Yes', 'No'], [yes, no], color=[sdg_colors[qid]] * 2)
                for j, val in enumerate([yes, no]):
                    ax.text(j, val + 1, str(val), ha='center')
                ax.set_ylim(0, max(yes, no) + 10)
                st.pyplot(fig)

# Section: Questionnaire Responses
elif section == "Questionnaire Responses":
    st.title("Completed Questionnaire Responses by Volunteer (Descending Order)")
    completed = questionnaire_df.groupby('volunteer_id')['membership_id'].nunique().reset_index(name='completed')
    completed = completed.sort_values(by='completed', ascending=False)

    fig, ax = plt.subplots()
    ax.bar(completed['volunteer_id'].astype(str), completed['completed'], color='#F5A623')
    for i, val in enumerate(completed['completed']):
        ax.text(i, val + 0.5, str(val), ha='center')
    ax.set_title("Completed Questionnaire Responses by Volunteer")
    ax.set_ylabel("Count")
    ax.set_xlabel("Volunteer ID")
    st.pyplot(fig)
