import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------
# NSGA-II Helper Functions
# --------------------------
def dominates(obj1, obj2):
    return (obj1[0] <= obj2[0] and obj1[1] >= obj2[1]) and (obj1[0] < obj2[0] or obj1[1] > obj2[1])

def non_dominated_sort(pop_objs):
    S, n, rank = {}, {}, {}
    fronts = [[]]
    for p in range(len(pop_objs)):
        S[p], n[p] = [], 0
        for q in range(len(pop_objs)):
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1]

# --------------------------
# NSGA-II for VM Bundles
# --------------------------
def nsga2_bundle_selection_safe(normalized_df, k=5, generations=50, pop_size=30, mutation_rate=0.2):
    n = len(normalized_df)

    def valid_cloud_distribution(sol):
        clouds = normalized_df.iloc[list(sol)]['CloudProvider']
        return clouds.value_counts().max() <= 2

    population = set()
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 20:
        candidate = tuple(sorted(random.sample(range(n), k)))
        if valid_cloud_distribution(candidate):
            population.add(candidate)
        attempts += 1
    population = list(population)

    def evaluate(sol):
        rows = normalized_df.iloc[list(sol)]
        cost = rows['MonthlyUSD'].sum()
        perf = (
            (1 - rows['Normalized_CPU Utilization (Average)']).mean() +
            (1 - rows['Normalized_Network In (Sum)']).mean() +
            (1 - rows['Normalized_Network Out (Sum)']).mean() +
            (1 - rows['Normalized_Disk Read Bytes (Sum)']).mean() +
            (1 - rows['Normalized_Disk Write Bytes (Sum)']).mean()
        ) / 5.0
        return [cost, perf]

    for _ in range(generations):
        objs = [evaluate(ind) for ind in population]
        fronts = non_dominated_sort(objs)

        selected = []
        for front in fronts:
            for idx in front:
                selected.append(population[idx])
                if len(selected) >= pop_size // 2:
                    break
            if len(selected) >= pop_size // 2:
                break

        offspring = set()
        while len(offspring) < pop_size - len(selected):
            p1, p2 = random.sample(selected, 2)
            cut = random.randint(1, k-1)
            child = list(p1[:cut]) + [g for g in p2 if g not in p1[:cut]]

            child = list(dict.fromkeys(child))
            while len(child) < k:
                candidate = random.randint(0, n-1)
                if candidate not in child:
                    child.append(candidate)
            child = child[:k]

            if random.random() < mutation_rate:
                replace_idx = random.randint(0, k-1)
                new_gene = random.randint(0, n-1)
                if new_gene not in child:
                    child[replace_idx] = new_gene

            while True:
                counts = normalized_df.iloc[child]['CloudProvider'].value_counts()
                if counts.max() <= 2:
                    break
                over_cloud = counts.idxmax()
                over_idx = [i for i, idx in enumerate(child)
                            if normalized_df.iloc[idx]['CloudProvider'] == over_cloud]
                replace_idx = random.choice(over_idx)
                candidates = [i for i in range(n)
                              if normalized_df.iloc[i]['CloudProvider'] != over_cloud and i not in child]
                if not candidates:
                    break
                child[replace_idx] = random.choice(candidates)

            child_tuple = tuple(sorted(child))
            if valid_cloud_distribution(child_tuple):
                offspring.add(child_tuple)

        combined = list(set(selected) | offspring)
        population = combined[:pop_size]

    objs = [evaluate(ind) for ind in population]
    fronts = non_dominated_sort(objs)

    unique_solutions = {}
    for idx in fronts[0]:
        sol_tuple = tuple(sorted(population[idx]))
        if sol_tuple not in unique_solutions:
            unique_solutions[sol_tuple] = objs[idx]

    results = []
    for sol_tuple, (cost, perf) in unique_solutions.items():
        results.append({'solution': sol_tuple, 'cost': cost, 'performance': perf})
    return results

# --------------------------
# Select Top 3 Recommendations
# --------------------------
def select_top_three(results):
    df = pd.DataFrame(results)
    best_cost = df.loc[df['cost'].idxmin()]
    best_perf = df.loc[df['performance'].idxmax()]

    remaining = df.drop([best_cost.name, best_perf.name])
    ideal_cost = df['cost'].min()
    ideal_perf = df['performance'].max()
    remaining['distance'] = np.sqrt((remaining['cost'] - ideal_cost)**2 + (remaining['performance'] - ideal_perf)**2)
    balanced = remaining.loc[remaining['distance'].idxmin()]

    return best_cost, best_perf, balanced, df

# --------------------------
# Streamlit App
# --------------------------
st.title("🌥️ Multi-Cloud VM Recommender (NSGA-II Optimized)")
st.markdown("""
This app finds **optimal 5-VM bundles** using **NSGA-II** considering:
- 💰 Minimized total cost  
- ⚡ Maximized average performance  
- ⚖️ Balanced cost-performance trade-off  
(Max 2 VMs per cloud provider)
""")

perf_file = st.file_uploader("Upload your normalized CSV (VM metrics)", type=["csv"])
price_file = st.file_uploader("Upload pricing CSV", type=["csv"])

if perf_file and price_file:
    perf_df = pd.read_csv(perf_file)
    price_df = pd.read_csv(price_file)
    perf_df.columns = perf_df.columns.str.strip()
    price_df.columns = price_df.columns.str.strip()

    # Show first 15 records
    st.subheader("📝 First 15 Records - Normalized VM Data")
    st.dataframe(perf_df.head(15))
    st.subheader("📝 First 15 Records - Pricing Data")
    st.dataframe(price_df.head(15))

    # Merge metrics and pricing
    merged_df = perf_df.merge(
        price_df[['InstanceName','Size','HourlyUSD','MonthlyUSD']],
        on=['InstanceName','Size'],
        how='inner'
    )

    metrics_cols = [
        'Normalized_CPU Utilization (Average)',
        'Normalized_Network In (Sum)',
        'Normalized_Network Out (Sum)',
        'Normalized_Disk Read Bytes (Sum)',
        'Normalized_Disk Write Bytes (Sum)'
    ]
    aggregated_df = merged_df.groupby(['CloudProvider','InstanceName','Size'], as_index=False)[metrics_cols + ['HourlyUSD','MonthlyUSD']].mean()
    
    # NSGA-II Optimization
    with st.spinner("Running NSGA-II optimization... ⏳"):
        results = nsga2_bundle_selection_safe(aggregated_df, k=5)
        low_cost, high_perf, balanced, df = select_top_three(results)

    # Top 3 Bundles
    st.subheader("🏆 Top 3 Recommended Bundles")
    def show_bundle(label, rec, color):
        sol_df = aggregated_df.iloc[list(rec['solution'])][[
            'CloudProvider','InstanceName','Size','HourlyUSD','MonthlyUSD'
        ]]
        st.markdown(f"### {label} Solution ({color})")
        st.write(f"**Total Monthly Cost:** ${rec['cost']:.2f}")
        st.write(f"**Avg Performance:** {rec['performance']:.4f}")
        st.dataframe(sol_df.reset_index(drop=True))

    show_bundle("💰 Low-Cost", low_cost, "🟩")
    show_bundle("⚡ High-Performance", high_perf, "🟥")
    show_bundle("⚖️ Balanced", balanced, "🟪")

    # --------------------------
    # Visualizations
    # --------------------------

    # Define CloudProvider colors
    clouds = aggregated_df['CloudProvider'].unique()
    colors = plt.cm.get_cmap('tab10', len(clouds))
    color_dict = {cloud: colors(i) for i, cloud in enumerate(clouds)}
    bar_colors = [color_dict[cp] for cp in aggregated_df['CloudProvider']]

    # VM Instances vs Monthly Pricing
    st.subheader("📊 VM Instances and Monthly Pricing")
    plt.figure(figsize=(14,6))
    plt.bar(aggregated_df['InstanceName'], aggregated_df['MonthlyUSD'], color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Monthly USD ($)")
    plt.title("VM Instances vs Monthly Pricing")
    handles = [plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=color_dict[c], markersize=10, label=c) for c in clouds]
    plt.legend(handles=handles, title="CloudProvider")
    st.pyplot(plt)

    # VM Instances vs Performance
    aggregated_df['AvgPerformance'] = 1 - aggregated_df[['Normalized_CPU Utilization (Average)',
                                                         'Normalized_Network In (Sum)',
                                                         'Normalized_Network Out (Sum)',
                                                         'Normalized_Disk Read Bytes (Sum)',
                                                         'Normalized_Disk Write Bytes (Sum)']].mean(axis=1)
    st.subheader("📊 VM Instances and Average Performance")
    plt.figure(figsize=(14,6))
    plt.bar(aggregated_df['InstanceName'], aggregated_df['AvgPerformance'], color=bar_colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Performance (Higher = Better)")
    plt.title("VM Instances vs Performance")
    plt.legend(handles=handles, title="CloudProvider")
    st.pyplot(plt)

    # Interactive Pricing vs Performance (Plotly)
    st.subheader("💵 Interactive Pricing vs Performance for All Instances")
    fig = px.scatter(
        aggregated_df,
        x='MonthlyUSD',
        y='AvgPerformance',
        color='CloudProvider',
        text='InstanceName',
        size=[20]*len(aggregated_df),
        hover_data={
            'InstanceName': True,
            'CloudProvider': True,
            'MonthlyUSD': True,
            'AvgPerformance': True
        },
        title="Pricing vs Performance for All Instances",
        labels={
            "MonthlyUSD": "Monthly Cost ($)",
            "AvgPerformance": "Average Performance"
        }
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(width=900, height=600, legend_title_text='CloudProvider')
    st.plotly_chart(fig, use_container_width=True)
