import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import re
import json

st.set_page_config(page_title="Dashboard", layout='wide')

# Sidebar navigation
st.sidebar.title("Graph Menu")
section = st.sidebar.radio("", [
    "Registrations Summary",
    "Age & Gender Insights",
    "SDG Security Questions",
    "Questionnaire Responses",
    "Model Output",
    "Model Output - Node Values",
    "Well Being Risk Indices",
    "Well Being Risk Categories - Primary Risk Factors"
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

def plot_and_download(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="jpeg", dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    st.download_button(
        label="Download as JPEG",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/jpeg"
    )

#Node_Id to Node_name mapping
nodeidnamemap_df = pd.read_excel("Nodeid_to_Nodename.xlsx")
# Clean the column names and contents
nodeidnamemap_df.columns = [col.strip() for col in nodeidnamemap_df.columns]
# Use correct column name now that we've verified them
nodeidnamemap_df["Node_name"] = nodeidnamemap_df["Node_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
nodeidnamemap_df["Node_id"] = nodeidnamemap_df["Node_id"].astype(str).str.strip()

# Create the mapping dictionary
node_id_to_name = dict(zip(nodeidnamemap_df["Node_id"], nodeidnamemap_df["Node_name"]))






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
    plot_and_download(fig, "registrations_summary.jpeg")

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
        plot_and_download(fig, "gender_total.jpeg")

    with col2:
        st.subheader("Gender Count by Age Group")
        bins = [0, 2, 14, 30, 60, float('inf')]
        labels = ['0-2', '2-14', '14-30', '30-60', '60+']
        family_df['age_group'] = pd.cut(family_df['age'], bins=bins, labels=labels, right=False)
        age_gender = family_df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)

        fig, ax = plt.subplots()
        age_gender.plot(kind='bar', ax=ax, color=['#4A90E2', '#F78DA7', '#A0A0A0'][:len(age_gender.columns)])
        plot_and_download(fig, "gender_by_age_group.jpeg")

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
                plot_and_download(fig, f"question_{qid}.jpeg")

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
    plot_and_download(fig, "questionnaire_responses.jpeg")

# Section: Model Output
elif section == "Model Output":
    st.title("Model Output - Node Statistics")

    import json

    # Load JSON from file structure
    try:
        with open('model_output.json') as f:
            data = json.load(f)
        
            content = f.read()

            print(content[52760:52810])  # Print around the error point
        # Extract summary statistics from each node
        stat_rows = []
        # Assuming you're interested in the first item (which has the 'results' key)
        for item in data[0].get('results', []):
            stats = item.get('summaryStatistics')
            if stats:
                stat_rows.append({
                    'node': item['node'],
                    'mean': stats.get('mean'),
                    'median': stats.get('median'),
                    'variance': stats.get('variance')
                })

        df = pd.DataFrame(stat_rows)

        if df.empty:
            st.warning("No summary statistics found in model_output.json.")
        else:
            metric = st.selectbox("Select Metric to Display", ['mean', 'median', 'variance'])
            selected_nodes = st.multiselect("Select up to 5 nodes", df['node'].tolist(), max_selections=5)

            filtered_df = df[df['node'].isin(selected_nodes)]

            if not filtered_df.empty:
                fig, ax = plt.subplots()
                ax.bar(filtered_df['node'], filtered_df[metric], color='teal')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f"{metric.capitalize()} Values of Selected Nodes")
                  # Add value labels
             #   ax.bar_label(bars, fmt='%.2f', padding=3)

               
                plot_and_download(fig, f"{metric}_by_node.jpeg")
            else:
                st.info("Please select 1 to 5 nodes to view the chart.")
    
    except FileNotFoundError:
        st.error("model_output.json not found in the directory.")

elif section == "Model Output - Node Values":
    st.title("Model Output - Node Result Values")

    import json

    try:
        with open('model_output.json') as f:
            data = json.load(f)

        # Collect all nodes with resultValues
        node_values = []
        for entry in data:
            for item in entry.get('results', []):
                if 'resultValues' in item:
                    node = item['node']
                    result_values = {rv['label']: rv['value'] for rv in item['resultValues']}
                    result_values['node'] = node
                    node_values.append(result_values)

        df_vals = pd.DataFrame(node_values)

        if df_vals.empty:
            st.warning("No resultValues data found.")
        else:
            selected_nodes = st.multiselect("Select up to 5 nodes", df_vals['node'].tolist(), max_selections=5)
            filtered_df = df_vals[df_vals['node'].isin(selected_nodes)]

            

            if not filtered_df.empty:
                plot_df = filtered_df.melt(id_vars='node', var_name='Category', value_name='Value')

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=plot_df, x='node', y='Value', hue='Category', ax=ax)

                ax.set_ylabel("Value")
                ax.set_xlabel("Node")
                ax.set_title("Result Values (Low, Medium, High) by Node")
                ax.legend(title="Category")
                
                plt.xticks(rotation=45)
    # Add value labels on top of each bar
                for bar in ax.patches:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                plot_and_download(fig, "node_result_values_grouped.jpeg")

            else:
                st.info("Please select 1 to 5 nodes to view their result values.")

    except FileNotFoundError:
        st.error("model_output.json not found.")


# Section: Well Being Risk Indices
elif section == "Well Being Risk Indices":
    st.title("Well Being Risk Indices")

    import json

    try:
        with open('model_results.json') as f:
            data = json.load(f)

        # Collect mean values from summaryStatistics for W_2 and all nodes like F4_1, G4_1, etc.
                # SDG color mapping
        sdg_colors = {
            "W_2": '#A0A0A0',
            "F4_1": '#E5243B',  # Food
            "I4_1": '#4C9F38',  # Infrastructure
            "H4_1": '#3F7E44',  # Health
            "E4_1": '#F36D25',  # Economic
            "C4_1": '#F89D2A',  # Environment
            "G4_1": '#C5192D',  # Gender
        }

        # Custom display order: W_2 first
        preferred_order = ["W_2", "F4_1", "I4_1", "H4_1", "E4_1", "C4_1", "G4_1"]

        summary_stats = []
        for entry in data:
            for result in entry.get("results", []):
                node = result.get("node")
                stats = result.get("summaryStatistics", {})
                mean_val = stats.get("mean")

                if node in preferred_order and mean_val is not None:
                    summary_stats.append({
                        "node": node,
                        "mean": round(mean_val * 100, 2),  # scale to percentage
                        "color": sdg_colors.get(node, "#4682B4")
                    })


        df_wellbeing = pd.DataFrame(summary_stats)
        df_wellbeing["node"] = pd.Categorical(df_wellbeing["node"], categories=preferred_order, ordered=True)
        df_wellbeing = df_wellbeing.sort_values("node")
        df_wellbeing["label"] = df_wellbeing["node"].astype(str).map(node_id_to_name).fillna(df_wellbeing["node"].astype(str))
      # Wrap long labels
        df_wellbeing["label"] = df_wellbeing["label"].apply(lambda x: "\n".join(x.split(" ", 3)))

        if df_wellbeing.empty:
            st.warning("No matching nodes (W_2 or *_4_*) found with mean values.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4.5))
           # bars = ax.bar(df_wellbeing["node"], df_wellbeing["mean"], color=df_wellbeing["color"])
            bars = ax.bar(df_wellbeing["label"], df_wellbeing["mean"], color=df_wellbeing["color"], width=0.6)



            ax.set_title("Well Being Risk Indices", fontsize=16)
            ax.set_ylabel("Mean", fontsize=12)
            ax.set_xlabel("Node", fontsize=7)
           
            plt.xticks(rotation=0, ha='center', fontsize=7)

            

            # Add value labels
            for bar in ax.patches:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

            plot_and_download(fig, "wellbeing_risk_indices_mean.jpeg")

    except FileNotFoundError:
        st.error("model_output.json not found.")
    except json.JSONDecodeError:
        st.error("Error decoding model_output.json. Please check the file format.")

# Section: Fourth-Level Node and Parent Means
elif section == "Well Being Risk Categories - Primary Risk Factors":
    st.title("Well Being Risk Categories - Primary Risk Factors")

    try:
        # Load JSON result data
        with open('model_results.json') as f:
            model_data = json.load(f)
        
        mean_values = {}
        for entry in model_data:
            for result in entry.get("results", []):
                node = result.get("node")
                mean = result.get("summaryStatistics", {}).get("mean")
                if node and mean is not None:
                    mean_values[node] = mean

        # Load node relationships
        relationship_df = pd.read_excel("Node_relationship.xlsx").dropna(subset=["parent", "child"])
        fourth_level_nodes = [c for c in relationship_df["child"].unique() if re.fullmatch(r"[A-Z]4_\d+", str(c))]
        parent_map = relationship_df.groupby("child")["parent"].apply(list).to_dict()

        # Display 3 charts per row
        row = st.columns(3)
        col_index = 0
         
        custom_titles = {
            "F4_1": "Risks to Food Security",
            "I4_1": "Risks to Infrastructure Security ",
            "H4_1": "Risks to Health Security",
            "E4_1": "Risks to Economic Security",
            "C4_1": "Risks to Environment and Climate Security",
            "G4_1": "Risks to Gender and Equality Security"
        }

        sdg_colors = {
            "F4_1": '#E5243B',  # Food
            "I4_1": '#4C9F38',  # Infrastructure
            "H4_1": '#3F7E44',  # Health
            "E4_1": '#F36D25',  # Economic
            "C4_1": '#F89D2A',  # Environment
            "G4_1": '#C5192D'   # Gender
        }
      # Prepare layout
        row = st.columns(3)
        col_index = 0

        for node in fourth_level_nodes:
            parents = parent_map.get(node, [])
            nodes = [node] + parents
            df = pd.DataFrame({
                "node": nodes,
                "label": [node_id_to_name.get(n, n) for n in nodes],
                "mean": [round(mean_values.get(n, 0) * 100, 2) for n in nodes]
            })

            # Wrap long labels
            df["label"] = df["label"].apply(lambda x: "\n".join(x.split(" ", 3)))

            if not df.empty:
                sdg_color = sdg_colors.get(node, "#4682B4")
                palette = [sdg_color] * len(nodes)

                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                sns.barplot(data=df, x="label", y="mean", ax=ax, palette=palette)

                ax.set_title(custom_titles.get(node, f"{node} and Parent Nodes"), fontsize=10)
                ax.set_ylabel("Mean")
                ax.set_xlabel("")
                # Add margin above tallest bar
                y_max = df["mean"].max()
                ax.set_ylim(0, y_max + 10)

                for bar in ax.patches:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

                # Display in current column
                with row[col_index]:
                    st.pyplot(fig)
                    buf = BytesIO()
                    fig.savefig(buf, format="jpeg", dpi=300, bbox_inches='tight')
                    st.download_button(
                        label="Download as JPEG",
                        data=buf.getvalue(),
                        file_name=f"{node}_mean_plot.jpeg",
                        mime="image/jpeg"
                    )

                # Move to next column
                col_index += 1
                if col_index == 3:
                    row = st.columns(3)
                    col_index = 0

    except Exception as e:
        st.error(f"Failed to load or render charts: {e}")
