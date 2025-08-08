import warnings
import os
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=DeprecationWarning)
px.defaults.template = "plotly_white"


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# Get full path to current file
current_file = os.path.abspath(__file__)

# Step up from Dash_app → to Financial inclusion
dash_app_folder = os.path.dirname(current_file)
project_root = os.path.dirname(dash_app_folder)

# Correct path to your CSV
csv_path = os.path.join(project_root, "Data", "processed", "world data.xlsx")

# Read the CSV
df = pd.read_csv(csv_path)

df["Indicator value"] = df["Indicator value"].astype(str).str.replace('%', '', regex=False).astype(float)

# --- Trends Page Variables ---
key_indicators = [
    "Account (% age 15+)",
    "Financial institution account (% age 15+)",
    "Mobile money account (% age 15+)",
    "Made or received a digital payment (% age 15+)"
]
df_filtered = df[(df["Indicator"].isin(key_indicators)) & (df["Year"].isin([2011, 2014, 2017, 2021]))].copy()

# --- Demographics Page Variables ---
dfs_keywords = [
    "mobile money", "digital payment", "internet", "online", "send money",
    "receive money", "used a mobile phone", "used the internet",
    "e-wallet", "used a mobile", "credit", "debit"
]

def assign_type(indicator):
    if pd.isna(indicator):
        return "Other"
    indicator_lower = indicator.lower()
    if "account" in indicator_lower:
        return "Account"
    if any(k in indicator_lower for k in dfs_keywords):
        return "Digital Financial Service Usage"
    return "Other"

# Apply the classification
df["Type"] = df["Indicator"].apply(assign_type)

# Filter the data
account_df = df[df["Type"] == "Account"]
dfs_df = df[df["Type"] == "Digital Financial Service Usage"]

# Create copies and clean indicators by removing text in parentheses
new_account = account_df.copy()
new_dfs = dfs_df.copy()
new_account["Indicator"] = new_account["Indicator"].str.replace(r"\s*\(.*?\)", "", regex=True)
new_dfs["Indicator"] = new_dfs["Indicator"].str.replace(r"\s*\(.*?\)", "", regex=True)

# Further breakdown by demographic group (all values as DataFrames — corrected)
account_data = {
    "Age": account_df[account_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": new_account[new_account["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": new_account[new_account["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": new_account[new_account["Indicator"].str.contains("Income", case=False, na=False)],
}

dfs_data = {
    "Age": dfs_df[dfs_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": new_dfs[new_dfs["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": new_dfs[new_dfs["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": new_dfs[new_dfs["Indicator"].str.contains("Income", case=False, na=False)],
}


# --- Clean Barrier Indicator ---
def clean_barrier_indicator(indicator):
    if pd.isna(indicator) or str(indicator).strip() == "":
        return None

    text = str(indicator).lower()

    # Remove suffixes like (% age 15+)
    for suffix in [
        "(% without an account 15 +)", "(% without an account age 15+)",
        "(% age 15+ without an account)", "(% age 15+)"
    ]:
        text = text.replace(suffix, "")

    text = text.replace("because", "")
    text = text.replace("due to", "")
    text = text.replace("reason for", "")
    text = text.strip()

    replacements = {
        "insufficient funds": "Insufficient Funds",
        "lack of money": "Lack of Money",
        "too far": "Too Far from Institution",
        "too expensive": "Too Expensive",
        "lack of documentation": "No ID / Documentation",
        "lack of trust": "Distrust Institutions",
        "religious reasons": "Religious Reasons",
        "religion": "Religious Reasons",
        "family member has account": "Family Member Has Account",
        "family member has": "Family Member Has Account",
        "do not need": "No Need",
        "not useful": "Not Useful",
        "no need": "No Need",
        "do not trust financial institutions": "Distrust Institutions",
        "do not trust": "Distrust Institutions",
        "no account": "No Account"
    }

    for phrase, short in replacements.items():
        if phrase in text:
            return short

    return text.title()

# --- Barriers Page Variables ---
barrier_keywords = (
    "barrier|reason for not having an account|no account|lack of money|too expensive|too far|"
    "family member has|lack of documentation|lack of trust|religious|no need|not useful|"
    "do not trust financial institutions"
)

df_barriers = df[df["Indicator"].str.contains(barrier_keywords, case=False, regex=True)].copy()
df_barriers["Clean Indicator"] = df_barriers["Indicator"].apply(clean_barrier_indicator)
df_barriers = df_barriers[df_barriers["Clean Indicator"].notna() & (df_barriers["Clean Indicator"] != "")]
df_barriers = df_barriers[df_barriers["Indicator value"].notna() & (df_barriers["Indicator value"] != "")]

# --- Clustering Function ---
def segment_countries_financial_inclusion(df, year, region=None, n_clusters=4):
    indicators = [
        "Account (% age 15+)",
        "Mobile money account (% age 15+)",
        "Made or received a digital payment (% age 15+)",
        "Saved at a financial institution (% age 15+)",
        "Borrowed from a financial institution (% age 15+)"
    ]
    df_year = df[(df['Year'] == year) & (df['Indicator'].isin(indicators))]
    if region:
        df_year = df_year[df_year['Region'] == region]

    df_pivot = df_year.pivot_table(index='Country name', columns='Indicator', values='Indicator value')
    if df_pivot.empty or df_pivot.shape[0] < n_clusters:
        return pd.DataFrame()

    df_pivot = df_pivot.dropna(thresh=int(0.5 * len(indicators)), axis=0)
    df_pivot.fillna(df_pivot.mean(), inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_pivot)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(pca_result)

    df_clustered = pd.DataFrame({
        'Country name': df_pivot.index,
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Segment': clusters
    })

    return df_clustered

# --- Layout ---
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand='Financial Inclusion and Digital Payment Adoption Dashboard',
        color='transparent',
        dark=True,
        className="mb-4"
    ),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Tabs(
                id="tabs",
                value='tab-home',
                children=[
                    dcc.Tab(label='Home', value='tab-home'),
                    dcc.Tab(label='Trends', value='tab-trends'),
                    dcc.Tab(label='Demographics', value='tab-demographics'),
                    dcc.Tab(label='Barriers', value='tab-barriers'),
                    dcc.Tab(label='Market Segmentations', value='tab-clusters'),
                ],
                vertical=True
            )
        ]), width=2),
        dbc.Col(html.Div(id="tabs-content", style={"padding": "16px"}), width=10)
    ])
], fluid=True)

# --- Tab Content Callback ---
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-home':
        return html.Div(className="home-container", children=[
            html.H1("Welcome to the Global Financial Inclusion Dashboard"),
            html.P("Gain powerful insights into how adults around the world access, manage, and use financial services from bank accounts to mobile money and digital payments."),
            html.H2("What This Dashboard Offers"),
            html.Ul([
                html.Li("Explore global and regional trends in financial account ownership"),
                html.Li("Analyze digital payment adoption across countries"),
                html.Li("Dive into demographic insights — impact of gender, income, education, and region"),
                html.Li("Understand barriers to financial inclusion in low- and middle-income countries"),
                html.Li("Visualize country clusters based on inclusion metrics")
            ]),
            html.H2("Why You Should Explore"),
            html.P("Financial inclusion is more than access — it’s a pathway to empowerment, opportunity, and resilience."),
            html.P("This dashboard uses real-world data to help:"),
            html.Ul([
                html.Li("Policy makers make informed decisions"),
                html.Li("Researchers identify key patterns and challenges"),
                html.Li("Innovators tailor solutions for underserved populations"),
            ])
        ])

    elif tab == 'tab-trends':
        return dbc.Container([
            html.H4("📈 Trends in Financial Inclusion", className="section-title"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='indicator-dropdown',
                    options=[{"label": ind, "value": ind} for ind in key_indicators],
                    value="Account (% age 15+)", clearable=False,
                    className="custom-dropdown"
                ), md=6),
                dbc.Col(dcc.RadioItems(
                    id='view-toggle',
                    options=[{"label": v, "value": v} for v in ["Global", "Regional", "Rwanda"]],
                    value="Global",
                    labelStyle={"display": "inline-block", "margin-right": "10px"},
                    className="custom-radio"
                ), md=6),
            ], className="mb-3"),
            dcc.Graph(id='combined-trend')
        ])
    elif tab == 'tab-demographics':
        return dbc.Container([
            html.H4("Demographics"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='data-type',
                    options=[{"label": "Financial Account Ownership ", "value": "Account"},
                             {"label": "Digital Financial Service Usage", "value": "Digital Financial Service Usage"}],
                    value="Account", clearable=False
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='demographic',
                    options=[{"label": f, "value": f} for f in ["Age", "Gender", "Education", "Income"]],
                    value="Age", clearable=False
                ), md=6),
            ]),
            dcc.Graph(id='bar-graph')
        ])
    elif tab == 'tab-barriers':
        return dbc.Container([
            html.H4("Barriers to Financial Inclusion"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='region-dropdown',
                    options=[{"label": r, "value": r} for r in sorted(df_barriers["Region"].dropna().unique())],
                    placeholder="Select Region"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='income-dropdown',
                    options=[{"label": i, "value": i} for i in sorted(df_barriers["Income group"].dropna().unique())],
                    placeholder="Select Income Group"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='country-dropdown',
                    options=[{"label": c, "value": c} for c in sorted(df_barriers["Country name"].dropna().unique())],
                    placeholder="Select Country"
                ), md=4),
            ]),
            dcc.Graph(id='barrier-graph')
        ])
    elif tab == 'tab-clusters':
        return dbc.Container([
            html.H4("Market Segmentations"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='year_dropdown',
                    options=[{'label': y, 'value': y} for y in sorted(df['Year'].unique())],
                    value=df['Year'].max()
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='country_dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df['Country name'].unique())],
                    placeholder='Highlight a Country (optional)', clearable=True
                ), md=6),
            ]),
            html.H5("1. Country Segmentation Map", className="menu-title", style={"marginTop": "20px"}),
            dcc.Graph(id='cluster_map', style={"height": "600px"}),

            html.H5("2. PCA-Based Country Clusters", className="menu-title", style={"marginTop": "30px"}),
            dcc.Graph(id='pca_scatter', style={"height": "600px"})
        ])


# --- Trend Callback ---
@app.callback(
    Output("combined-trend", "figure"),
    [Input("indicator-dropdown", "value"),
     Input("view-toggle", "value")]
)
def update_combined_trend(indicator, view):
    trend_data = []
    if view == "Global":
        global_df = df_filtered[df_filtered["Indicator"] == indicator].groupby("Year")["Indicator value"].mean().reset_index()
        global_df["Source"] = "Global Average"
        trend_data.append(global_df)
        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)
    elif view == "Regional":
        region_df = df_filtered[df_filtered["Indicator"] == indicator].groupby(["Region", "Year"])["Indicator value"].mean().reset_index()
        region_df.rename(columns={"Region": "Source"}, inplace=True)
        trend_data.append(region_df)
    elif view == "Rwanda":
        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)
    combined_df = pd.concat(trend_data)
    fig = px.line(combined_df, x="Year", y="Indicator value", color="Source", markers=True)
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, family='Lato, sans-serif', color='#3490ac', weight='bold')
    )
    return fig

# --- Demographics Callback ---
@app.callback(
    Output("bar-graph", "figure"),
    [Input("data-type", "value"),
     Input("demographic", "value")]
)
def update_demographic_graph(data_type, demographic):
    df_selected = account_data.get(demographic) if data_type == "Account" else dfs_data.get(demographic)
    if df_selected is None or df_selected.empty:
        fig = px.bar(title="No data available.")
    else:
        df_summary = (
            df_selected.groupby("Indicator")["Indicator value"]
            .mean()
            .reset_index()
            .sort_values(by="Indicator value", ascending=True)
            .head(20)
        )
        fig = px.bar(df_summary, x="Indicator value", y="Indicator", orientation="h", color_discrete_sequence=["#3490ac"])
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, family='Lato, sans-serif', color='#3490ac', weight='bold')
    )
    return fig

# --- Barriers Callback ---
@app.callback(
    Output("barrier-graph", "figure"),
    [Input("region-dropdown", "value"),
     Input("income-dropdown", "value"),
     Input("country-dropdown", "value")]
)
def update_barrier_graph(region, income, country):
    filtered = df_barriers.copy()
    if region:
        filtered = filtered[filtered["Region"] == region]
    if income:
        filtered = filtered[filtered["Income group"] == income]
    if country:
        filtered = filtered[filtered["Country name"] == country]
    if filtered.empty:
        fig = px.bar(title="No data available.")
    else:
        summary = filtered.groupby("Clean Indicator")["Indicator value"].mean().reset_index()
        fig = px.bar(
            summary.sort_values(by="Indicator value", ascending=True),
            x="Indicator value",
            y="Clean Indicator",
            orientation="h",
            color_discrete_sequence=["#3490ac"],
            labels={"Indicator value": "Average %", "Clean Indicator": "Barrier"}
        )
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, family='Lato, sans-serif', color='#3490ac', weight='bold')
    )
    return fig

# --- Cluster Callbacks ---
@app.callback(
    Output('cluster_map', 'figure'),
    [Input('year_dropdown', 'value'),
     Input('country_dropdown', 'value')]  # added country input
)
def update_cluster_choropleth(selected_year, selected_country):
    df_segmented = segment_countries_financial_inclusion(df, selected_year)
    if df_segmented.empty:
        fig = px.choropleth(title="Not enough data to cluster.")
        return fig

    segment_names = {
        0: "Basic Access",
        1: "Developing Access",
        2: "Growing Access",
        3: "Advanced Access"
    }
    color_map = {
        "Basic Access": "#276c88",
        "Developing Access": "#3490ac",
        "Growing Access": "#4eb2c0",
        "Advanced Access": "#7dd6e8"
    }

    df_segmented["Segment Label"] = df_segmented["Segment"].map(segment_names)

    if selected_country:
        selected_segment = df_segmented.loc[df_segmented["Country name"] == selected_country, "Segment Label"]
        if not selected_segment.empty:
            selected_segment = selected_segment.iloc[0]
            df_segmented = df_segmented[df_segmented["Segment Label"] == selected_segment]

    fig = px.choropleth(
        df_segmented,
        locations="Country name",
        locationmode='country names',
        color="Segment Label",
        hover_name="Country name",
        color_discrete_map=color_map,
        category_orders={"Segment Label": list(segment_names.values())}
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="RebeccaPurple")

    fig.update_layout(
        title=f"Market Segmentation by Financial Inclusion - Year: {selected_year}",
        title_x=0.5,
        title_font=dict(size=20, family='Lato, sans-serif', color='#3490ac', weight='bold'),
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    return fig


@app.callback(
    Output('pca_scatter', 'figure'),
    [Input('year_dropdown', 'value'),
     Input('country_dropdown', 'value')]
)
def update_cluster_scatter(selected_year, selected_country):
    df_segmented = segment_countries_financial_inclusion(df, selected_year)
    if df_segmented.empty:
        fig = px.scatter(title="Not enough data to cluster.")
        return fig

    segment_names = {
        0: "Basic Access",
        1: "Developing Access",
        2: "Growing Access",
        3: "Advanced Access"
    }
    color_map = {
        "Basic Access": "#276c88",
        "Developing Access": "#3490ac",
        "Growing Access": "#4eb2c0",
        "Advanced Access": "#7dd6e8"\
    }

    df_segmented["Segment Label"] = df_segmented["Segment"].map(segment_names)

    if selected_country:
        selected_segment = df_segmented.loc[df_segmented["Country name"] == selected_country, "Segment Label"]
        if not selected_segment.empty:
            selected_segment = selected_segment.iloc[0]
            df_segmented = df_segmented[df_segmented["Segment Label"] == selected_segment]

    fig = px.scatter(
        df_segmented,
        x="PCA1",
        y="PCA2",
        color="Segment Label",
        hover_name="Country name",
        color_discrete_map=color_map
    )

    if selected_country:
        selected_row = df_segmented[df_segmented["Country name"] == selected_country]
        if not selected_row.empty:
            fig.add_scatter(
                x=selected_row["PCA1"],
                y=selected_row["PCA2"],
                mode='markers+text',
                text=selected_country,
                marker=dict(size=12, color='black', symbol='star'),
                name='Selected Country'
            )

    plot_title = f"PCA Scatter of Country Clusters - Year: {selected_year}"
    if selected_country:
        plot_title += f", Highlight: {selected_country}"

    fig.update_layout(
        title=plot_title,
        title_x=0.5,
        title_font=dict(size=20, family='Lato, sans-serif', color='#3490ac', weight='bold')
    )
    return fig
@app.callback(
    Output('year_dropdown', 'options'),
    Output('year_dropdown', 'value'),
    Input('country_dropdown', 'value')
)
def update_year_options(selected_country):
    if selected_country:
        # Filter data for selected country
        years_for_country = sorted(df[df['Country name'] == selected_country]['Year'].unique())
        if years_for_country:
            options = [{'label': y, 'value': y} for y in years_for_country]
            value = max(years_for_country)
            return options, value
        else:
            # fallback to all years if no data found for country
            all_years = sorted(df['Year'].unique())
            options = [{'label': y, 'value': y} for y in all_years]
            return options, max(all_years)
    else:
        # No country selected: all years
        all_years = sorted(df['Year'].unique())
        options = [{'label': y, 'value': y} for y in all_years]
        return options, max(all_years)

# --- Run Server ---
if __name__ == "__main__":
    app.run(debug=True, port=8078)