
'''
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
'''
