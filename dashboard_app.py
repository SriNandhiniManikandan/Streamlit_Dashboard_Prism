import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import re
import json
import textwrap
import numpy as np

st.set_page_config(page_title="Dashboard", layout='wide')

# Sidebar navigation
st.sidebar.title("Analytics and Risk Assessment")
section = st.sidebar.radio("", [
    "Registration Summary",
    "Demographic Analysis",
    "Well Being Perceptions",
    #"Questionnaire Responses",
  #  "Model Output - Node Values",
    "Well Being Risk Indices",
    "Risks to Food Security",
    "Risks to Health Security",
    "Risks to Gender Equality",
    "Risks to Economic Security",
    "Risks to Infrastructure Security",
    "Risks to Environment and Climate Security",
    "Risk Comparison Options"
  #  "Well Being Risk Categories - Primary Risk Factors",
  #  "Well Being Risk Categories - Primary Risk Factors 2",

])







# Cached data loading and preprocessing
@st.cache_data
def load_data():
    household_df = pd.read_excel("Household_Details_14052025_to_26062025.xlsx")
    family_df = pd.read_excel("Household_Family_Members_14052025_to_26062025.xlsx")
    questionnaire_df = pd.read_excel("Most_Recent_Questionnaire_Responses_till_26062025.xlsx")
    
  
    # Shared preprocessing
    household_df['created_at'] = pd.to_datetime(household_df['created_at'])
    household_df['date'] = household_df['created_at'].dt.date
    family_df['age'] = pd.to_numeric(family_df['age'], errors='coerce')

    return household_df, family_df, questionnaire_df

# Load data
household_df, family_df, questionnaire_df = load_data()

def wrap_text(label, width=18):

    return "\n".join(textwrap.wrap(label, width))

#Node_Id to Node_name mapping
nodeidnamemap_df = pd.read_excel("Nodeid_to_Nodename.xlsx")
# Clean the column names and contents
nodeidnamemap_df.columns = [col.strip() for col in nodeidnamemap_df.columns]
# Use correct column name now that we've verified them
nodeidnamemap_df["Node_name"] = nodeidnamemap_df["Node_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
nodeidnamemap_df["Node_id"] = nodeidnamemap_df["Node_id"].astype(str).str.strip()

# Create the mapping dictionary
node_id_to_name = dict(zip(nodeidnamemap_df["Node_id"], nodeidnamemap_df["Node_name"]))



def render_node_charts(category_prefix, title, fourth_level_customs, third_level_customs, sdg_color):
    st.header(f"{title}")

    # Load model results
    with open("model_results.json") as f:
        model_data = json.load(f)

    yes_no_values = {}
    for entry in model_data:
        for result in entry.get("results", []):
            node = result.get("node")
            values = result.get("resultValues", [])
            counts = {v["label"]: round(v["value"]*100,2)for v in values if v["label"] in ["Yes", "No"]}
            if node and counts:
                yes_no_values[node] = counts
    mean_values = {}
    for entry in model_data:
        for result in entry.get("results", []):
            node = result.get("node")
            mean = result.get("summaryStatistics", {}).get("mean")
            if node and mean is not None:
                mean_values[node] = mean

    # Load node relationships
    relationship_df = pd.read_excel("Node_relationship.xlsx").dropna(subset=["parent", "child"])
    relationship_df["parent"] = relationship_df["parent"].str.strip()
    relationship_df["child"] = relationship_df["child"].str.strip()
    parent_map = relationship_df.groupby("child")["parent"].apply(list).to_dict()

    # Utility: wraps long labels
    def wrap_text(label, max_words=4):
        parts = label.split(" ")
        return "\n".join([" ".join(parts[i:i + max_words]) for i in range(0, len(parts), max_words)])

    # --- PRIMARY RISK INDICATORS (Level 4) ---
    st.subheader("Primary Risk Indicators")
    level4_nodes = [n for n in relationship_df["child"].unique() if re.fullmatch(rf"{category_prefix}4_\d+", str(n))]
    row = st.columns(2)
    col_index = 0

    for node in level4_nodes:
        parents = parent_map.get(node, [])
        nodes = [node] + parents
        df = pd.DataFrame({
            "node": nodes,
            "label": [node_id_to_name.get(n, n) for n in nodes],
            "mean": [round(mean_values.get(n, 0) * 100, 2) for n in nodes]
        })
        df["label"] = df["label"].apply(wrap_text)

        if not df.empty:
            child_label = df["label"].iloc[0]
            child_value = df["mean"].iloc[0]
            parent_labels = df["label"].iloc[1:]
            parent_values = df["mean"].iloc[1:]

            palette = [sdg_color] * len(parent_labels)
            fig, ax = plt.subplots(figsize=(7, 5))
           # ax.set_xticklabels(parent_labels, rotation=15, ha='right', fontsize=9)

            bars = ax.bar(parent_labels, parent_values, color=palette, width=0.5)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

            # Plot line for child
          # Plot dashed line (no marker)
            line_label = f"{child_label}"
            ax.axhline(y=child_value, color='black', linestyle='--', linewidth=2, label=line_label)

            # Annotate value in center of plot
            center_x = len(parent_labels) / 2 - 0.5  # Center of the bar group
            ax.annotate(f'{child_value:.2f}', xy=(center_x, child_value), xytext=(0, 3),
                        textcoords="offset points", ha='center', fontsize=9)
            ax.set_xticklabels(parent_labels, rotation=15, ha='right', fontsize=9)
            ax.set_title(child_label, fontsize=11)
            ax.set_ylabel("Mean")
            ax.set_xlabel("")
            ax.set_ylim(0, max([child_value] + parent_values.tolist()) + 10)
            ax.legend(loc='upper right')

            with row[col_index]:
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="jpeg", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="Download as JPEG",
                    data=buf.getvalue(),
                    file_name=f"{node}_combined_chart.jpeg",
                    mime="image/jpeg"
                )

            col_index += 1
            if col_index == 2:
                row = st.columns(2)
                col_index = 0

    # --- SECONDARY RISK INDICATORS (Level 3) ---
    st.subheader("Secondary Risk Indicators")
    level3_nodes = [n for n in relationship_df["child"].unique() if re.fullmatch(rf"{category_prefix}3_\d+", str(n))]
    row = st.columns(2)
    col_index = 0

    for node in level3_nodes:
        parents = parent_map.get(node, [])
        nodes = [node] + parents
        df_3 = pd.DataFrame({
            "node": nodes,
            "label": [node_id_to_name.get(n, n) for n in nodes],
            "mean": [round(mean_values.get(n, 0) * 100, 2) for n in nodes]
        })
        df_3["label"] = df_3["label"].apply(wrap_text)

        if not df_3.empty:
            child_label = df_3["label"].iloc[0]
            child_value = df_3["mean"].iloc[0]
            parent_labels = df_3["label"].iloc[1:]
            parent_values = df_3["mean"].iloc[1:]

            palette = [sdg_color] * len(parent_labels)
            fig, ax = plt.subplots(figsize=(7, 5))

            bars = ax.bar(parent_labels, parent_values, color=palette, width=0.5)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

           # ax.plot([0], [child_value], marker='o', color='black')
            # Plot dashed line (no marker)
            line_label = f"{child_label}"
            ax.axhline(y=child_value, color='black', linestyle='--', linewidth=2, label=line_label)

            # Annotate value in center of plot
            center_x = len(parent_labels) / 2 - 0.5  # Center of the bar group
            ax.annotate(f'{child_value:.2f}', xy=(center_x, child_value), xytext=(0, 3),
                        textcoords="offset points", ha='center', fontsize=9)
          #  ax.axhline(y=child_value, color='black', linestyle='--', linewidth=2, label=child_label)
           # ax.annotate(f'{child_value:.2f}', xy=(0, child_value), xytext=(0, 3),
                #        textcoords="offset points", ha='center', va='bottom')
            ax.set_xticklabels(parent_labels, rotation=15, ha='right', fontsize=9)
            ax.set_title(child_label, fontsize=11)
            ax.set_ylabel("Mean")
            ax.set_xlabel("")
            ax.set_ylim(0, max([child_value] + parent_values.tolist()) + 10)
            ax.legend(loc='upper right', fontsize=9) 
       
            with row[col_index]:
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="jpeg", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="Download as JPEG",
                    data=buf.getvalue(),
                    file_name=f"{node}_combined_chart.jpeg",
                    mime="image/jpeg"
                )

            col_index += 1
            if col_index == 2:
                row = st.columns(2)
                col_index = 0



    # -------- BASIC INDICATORS (Level 2) --------
    st.subheader("Basic Indicators")
    level2_nodes = [n for n in relationship_df["child"].unique() if re.fullmatch(rf"{category_prefix}2_\d+", str(n))]
    row = st.columns(2)
    col_index = 0

    for node in level2_nodes:
        parents = parent_map.get(node, [])
        valid_parents = [p for p in parents if p in yes_no_values]
        if not valid_parents:
            continue

        child_value = round(mean_values.get(node, 0) * 100, 2)
        child_label = node_id_to_name.get(node, node)
        child_label_wrapped = wrap_text(child_label)
        #child_label = wrap_text(node_id_to_name.get(node, node))

        bar_labels = []
        heights = []
        for parent in valid_parents:
            parent_name = node_id_to_name.get(parent, parent)
            yes_val = yes_no_values[parent]["Yes"] 
            no_val = yes_no_values[parent]["No"]
            bar_labels.extend([f"{wrap_text(parent_name)} - Yes", f"{wrap_text(parent_name)} - No"])
            heights.extend([yes_val, no_val])

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        x = np.arange(len(heights))
        bars = ax1.bar(x, heights, color=sdg_color, width=0.5)

        for i, bar in enumerate(bars):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}",
                    ha='center', va='bottom')
        line_label = f"{child_label_wrapped}"
        ax2.axhline(y=child_value, color='black', linestyle='--', linewidth=2,label=line_label)
        ax2.annotate(f"{child_value:.1f}", xy=(x.mean(), child_value), xytext=(0, 3),
                 textcoords="offset points", ha='center')
     #   ax2.text(len(heights) - 0.5, child_value + 2, f"{child_label}: {child_value:.1f}%", color='black', ha='right')

        ax1.set_xticks(x)
        ax1.set_xticklabels(bar_labels, rotation=30, ha='right')
        ax1.set_ylabel("Yes/No")
      #  ax2.set_ylabel("Mean (Child)")
        ax1.set_ylim(0, max(heights) + 20)
        ax2.set_ylim(0, max(heights + [child_value]) + 20)
        ax2.legend(loc='upper right', fontsize=9) 
        ax1.set_title(child_label)

        with row[col_index]:
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="jpeg", dpi=300, bbox_inches='tight')
            st.download_button(
                label="Download as JPEG",
                data=buf.getvalue(),
                file_name=f"{node}_basic_indicator.jpeg",
                mime="image/jpeg"
            )

        col_index = (col_index + 1) % 2
        if col_index == 0:
            row = st.columns(2)



# Utility for wrapping labels
def wrap_text(label, max_words=4):
    parts = label.split(" ")
    return "\n".join([" ".join(parts[i:i+max_words]) for i in range(0, len(parts), max_words)])


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







# Section: Registrations Summary
if section == "Registration Summary":
    st.title("Household Registration Summary")
    total = household_df.groupby('volunteer_id').size().reset_index(name='total')
    avg = household_df.groupby(['volunteer_id', 'date']).size().groupby('volunteer_id').mean().reset_index(name='avg')
    df = pd.merge(total, avg, on='volunteer_id')
    df = df[~df['volunteer_id'].isin([3, 11])]
    df = df.sort_values(by='total', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    bar1 = ax.bar([i - 0.2 for i in x], df['total'], width=0.4, label='Total', color='skyblue')
    bar2 = ax.bar([i + 0.2 for i in x], df['avg'], width=0.4, label='Avg/Day', color='orange')

    # Add value labels on top of each bar
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(df['volunteer_id'], rotation=45)
    ax.legend()
    ax.set_title("Total & Average Registrations per Volunteer")
    ax.set_xlabel("Volunteer ID")
    plot_and_download(fig, "registrations_summary.jpeg")

# Section: Age & Gender Insights
elif section == "Demographic Analysis":
    st.title("Demographic Distribution of Member Households")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gender Distribution of Member Households")
        gender_total = family_df['gender'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(gender_total.index, gender_total.values, color=['#4A90E2', '#F78DA7', '#A0A0A0'][:len(gender_total)])
        for i, val in enumerate(gender_total.values):
            ax.text(i, val + 1, str(val), ha='center')
        plot_and_download(fig, "gender_total.jpeg")

    with col2:
        st.subheader("Gender Distribution Age Group")
        bins = [0, 2, 14, 30, 60, float('inf')]
        labels = ['0-2', '2-14', '14-30', '30-60', '60+']
        family_df['age_group'] = pd.cut(family_df['age'], bins=bins, labels=labels, right=False)
        age_gender = family_df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)

        fig, ax = plt.subplots()
        age_gender.plot(kind='bar', ax=ax, color=['#4A90E2', '#F78DA7', '#A0A0A0'][:len(age_gender.columns)])
        plot_and_download(fig, "gender_by_age_group.jpeg")

# Section: SDG Security Questions
elif section == "Well Being Perceptions":
    st.title("Initial Perceptions on Well-Being Attributes")
    question_texts = {
        91: "Food Security",
        92: "Health Security",
        93: "Environment and Climate Security",
        94: "Economic Security",
        95: "Housing Security",
        96: "Gender Equality"
    }
    sdg_colors = {
        91: '#DDA63A', 92: '#4C9F38', 93: '#3F7E44',
        94: '#A21942', 95: '#FD9D24', 96: '#FF3A21'
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
                fig, ax = plt.subplots(figsize=(6.5, 5))
                ax.bar(['Yes', 'No'], [yes, no], color=[sdg_colors[qid]] * 2)
                for j, val in enumerate([yes, no]):
                    ax.text(j, val + 10, str(val), ha='center', va='bottom', fontsize=10)
                ax.set_ylim(0, max(yes, no) + 200)
                ax.set_ylabel("Count")
                plot_and_download(fig, f"question_{qid}.jpeg")


# Section: Model Output
elif section == "Risk Comparison Options":
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
 # Map node IDs to names for display
            node_name_map = {nid: node_id_to_name.get(nid, nid) for nid in df['node']}
            name_to_id_map = {v: k for k, v in node_name_map.items()}  # reverse mapping

            # Display dropdown with node names
            selected_names = st.multiselect(
                "Select up to 5 nodes",
                options=list(name_to_id_map.keys()),
                max_selections=5
            )


            selected_ids = [name_to_id_map[name] for name in selected_names]
            filtered_df = df[df['node'].isin(selected_ids)]

            if not filtered_df.empty:
                # Replace node IDs with readable names in plot
                filtered_df["label"] = filtered_df["node"].map(node_name_map)
                fig, ax = plt.subplots()
                bars = ax.bar(filtered_df['label'], filtered_df[metric], color='teal')

                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

                ax.set_ylabel(metric.capitalize())
                ax.set_title(f"{metric.capitalize()} Values of Selected Nodes")
                plot_and_download(fig, f"{metric}_by_node.jpeg")
            else:
                st.info("Please select 1 to 5 nodes to view the chart.")

    except FileNotFoundError:
        st.error("model_output.json not found in the directory.")

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
            "W_2": "#18A6EC",
            "F4_1": '#DDA63A',  # Food
            "I4_1": '#FD9D24',  # Infrastructure
            "H4_1": '#4C9F38',  # Health
            "E4_1": '#A21942',  # Economic
            "C4_1": '#3F7E44',  # Environment
            "G4_1": '#FF3A21',  # Gender
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

elif section == "Risks to Food Security":
    food_custom_4 = {
        "F4_1": "Risks to Food Security"
    }
    food_custom_3 = {
        "F3_1": "Risks to Food Accessibility and Nutrition",
        "F3_2": "Risks to Food Stability and Availability"
    }
    render_node_charts("F", "Food Security Category", food_custom_4, food_custom_3, '#DDA63A')

# Gender
elif section == "Risks to Gender Equality":
    gender_custom_4 = {
        "G4_1": "Risks to Gender and Equality Security"
    }
    gender_custom_3 = {
        "G3_1": "Gender Inequality in Education and Employment",
        "G3_2": "Gender Disparity in Health and Safety"
    }
    render_node_charts("G", "Gender Equality Category", gender_custom_4, gender_custom_3, '#FF3A21')

# Health
elif section == "Risks to Health Security":
    health_custom_4 = {
        "H4_1": "Risks to Health Security"
    }
    health_custom_3 = {
        "H3_1": "Barriers to Healthcare Access",
        "H3_2": "Healthcare Quality and Affordability Risks",
        "H3_3": "Mental Health and Emotional Well-being"
    }
    render_node_charts("H", "Health Security Category", health_custom_4, health_custom_3, '#4C9F38')

# Infrastructure
elif section == "Risks to Infrastructure Security":
    infra_custom_4 = {
        "I4_1": "Risks to Infrastructure Security"
    }
    infra_custom_3 = {
        "I3_1": "Infrastructure Reliability and Accessibility",
        "I3_2": "Technology Access and Digital Inclusion"
    }
    render_node_charts("I", "Infrastructure Security Category", infra_custom_4, infra_custom_3, '#FD9D24')

# Economic
elif section == "Risks to Economic Security":
    economic_custom_4 = {
        "E4_1": "Risks to Economic Security"
    }
    economic_custom_3 = {
        "E3_1": "Income and Employment Insecurity",
        "E3_2": "Access to Financial Services",
        "E3_3": "Cost of Living Pressures"
    }
    render_node_charts("E", "Economic Security Category", economic_custom_4, economic_custom_3, '#A21942')

# Environment and Climate
elif section == "Risks to Environment and Climate Security":
    climate_custom_4 = {
        "C4_1": "Risks to Environment and Climate Security"
    }
    climate_custom_3 = {
        "C3_1": "Environmental Degradation Risks",
        "C3_2": "Climate-Induced Displacement and Resource Scarcity"
    }
    render_node_charts("C", "Environment and Climate Security Category", climate_custom_4, climate_custom_3, '#3F7E44')








