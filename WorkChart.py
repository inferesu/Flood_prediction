import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD AND PROCESS THE CSV DATA
# ==========================================
# Replace 'your_data.csv' with the actual path to your file
file_path = '/Users/anastasija.safonova/Downloads/Release QA Dashboard_Comparative Bug Reporting Stats_Table.csv'

# Load the raw data
df_raw = pd.read_csv(file_path)

# Convert the "Date Created" column to actual datetime objects
# The format in your screenshot looks like "Mar 31, 2026"
df_raw['Date Created'] = pd.to_datetime(df_raw['Date Created'])

# Extract a sorting column (Period) and a display column (Month_Display)
df_raw['Period'] = df_raw['Date Created'].dt.to_period('M')
df_raw['Month'] = df_raw['Date Created'].dt.strftime('%b %Y')

# Count the occurrences of each Reporter Organization per month
# This creates a table where rows are Months, columns are Orgs, and values are counts
monthly_counts = df_raw.groupby(['Period', 'Month', 'Reporter Organization']).size().unstack(fill_value=0)

# Ensure the columns we need exist (in case a month is missing one entirely)
for col in ['External', 'Internal', 'RQA']:
    if col not in monthly_counts.columns:
        monthly_counts[col] = 0

# Calculate the percentage shares per month
monthly_totals = monthly_counts.sum(axis=1)
df = (monthly_counts.div(monthly_totals, axis=0) * 100).reset_index()

# Sort chronologically using the Period column, then drop it as we only need 'Month' for the chart
df = df.sort_values('Period').drop(columns=['Period'])

# ==========================================
# 2. SETUP THE CHART
# ==========================================
fig, ax = plt.subplots(figsize=(16, 6))

color_external = '#E2950B'  # Golden/Orange
color_internal = '#5DADE2'  # Light Blue
color_rqa = '#16A085'  # Teal/Green

# Plot the lines
ax.plot(df['Month'], df['External'], marker='o', markersize=5, color=color_external, label='External', linewidth=1.5)
ax.plot(df['Month'], df['Internal'], marker='o', markersize=5, color=color_internal, label='Internal', linewidth=1.5)
ax.plot(df['Month'], df['RQA'], marker='o', markersize=5, color=color_rqa, label='RQA', linewidth=1.5)

# ==========================================
# 3. ADD RQA PERCENTAGE LABELS
# ==========================================
for index, row in df.iterrows():
    label_text = f"{row['RQA']:.1f}%"

    # Calculate x position dynamically (since reset_index might mess up standard enumeration)
    x_pos = df.index.get_loc(index)

    ax.annotate(label_text,
                (x_pos, row['RQA']),
                textcoords="offset points",
                xytext=(0, 6),
                ha='center',
                fontsize=8,
                color='#333333')

# ==========================================
# 4. FORMATTING & STYLING
# ==========================================
ax.set_title('Monthly Share of Issues by Source + RQA % Labels', fontsize=16, color='#444', pad=15)
ax.set_ylabel('Share of Issues (%)', fontsize=12)

# Rotate X-axis labels 45 degrees
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# Set Y-axis limits slightly above 100 or highest value for breathing room
max_val = df[['External', 'Internal', 'RQA']].max().max()
ax.set_ylim(0, min(100, max_val + 10))

# Grid styling
ax.grid(True, linestyle='--', alpha=0.5, color='#B0B0B0')

# Remove top/right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')

# Legend
ax.legend(loc='upper right', framealpha=1)

plt.tight_layout()
plt.show()
