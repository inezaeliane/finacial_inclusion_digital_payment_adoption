import warnings
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load and prepare data
df = pd.read_excel("world data.xlsx")
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
    "receive money", "used a mobile phone", "used the internet", "e-wallet", "used a mobile"
]
def assign_type(indicator):
    if pd.isna(indicator):
        return "Account"
    indicator_lower = indicator.lower()
    return "DFS" if any(k in indicator_lower for k in dfs_keywords) else "Account"

df["Type"] = df["Indicator"].apply(assign_type)
account_df = df[df["Type"] == "Account"]
dfs_df = df[df["Type"] == "DFS"]

account_data = {
    "Age": account_df[account_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": account_df[account_df["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": account_df[account_df["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": account_df[account_df["Indicator"].str.contains("Income", case=False, na=False)],
}

dfs_data = {
    "Age": dfs_df[dfs_df["Indicator"].str.contains("Age", case=False, na=False)],
    "Gender": dfs_df[dfs_df["Indicator"].str.contains("female|male", case=False, na=False)],
    "Education": dfs_df[dfs_df["Indicator"].str.contains("Education", case=False, na=False)],
    "Income": dfs_df[dfs_df["Indicator"].str.contains("Income", case=False, na=False)],
}

# --- Barriers Page Variables ---
barrier_keywords = (
    "barrier|reason for not having an account|no account|lack of money|too expensive|too far|"
    "family member has|lack of documentation|lack of trust|religious|no need|not useful|"
    "do not trust financial institutions"
)
df_barriers = df[df["Indicator"].str.contains(barrier_keywords, case=False, regex=True)]

# --- Clustering Function ---
def segment_countries(df, year, region=None):
    df_filtered = df[df['Year'] == year]
    if region:
        df_filtered = df_filtered[df_filtered['Region'] == region]

    df_pivot = df_filtered.pivot_table(index='Country name', columns='Indicator', values='Indicator value')
    df_pivot = df_pivot.dropna(thresh=int(0.5 * len(df_pivot)), axis=1).fillna(df_pivot.mean())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pivot)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pivot = df_pivot.assign(PCA1=pca_result[:, 0], PCA2=pca_result[:, 1]).copy()
    df_pivot.reset_index(inplace=True)

    income_info = df_filtered[['Country name', 'Income group']].drop_duplicates()
    df_segmented = df_pivot.merge(income_info, on='Country name', how='left')
    df_segmented.rename(columns={'Income group': 'Segment'}, inplace=True)

    return df_segmented

# --- App Layout ---
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="📊 Financial Inclusion Dashboard",
        color="primary",
        dark=True,
        className="mb-4 navbar-custom"  # for CSS styling
    ),

    dcc.Tabs(id="tabs", value='tab-trends', className="tabs-container", children=[
        dcc.Tab(label='Trends', value='tab-trends', className="tab-item"),
        dcc.Tab(label='Demographics', value='tab-demographics', className="tab-item"),
        dcc.Tab(label='Barriers', value='tab-barriers', className="tab-item"),
        dcc.Tab(label='Clusters', value='tab-clusters', className="tab-item"),
    ]),
    html.Div(id='tabs-content', className="content-container")
], fluid=True)

# --- Render Tabs ---
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    graph_container = {"className": "graph-container", "style": {"height": "500px"}}

    if tab == 'tab-trends':
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
            dcc.Graph(id='combined-trend', **graph_container)
        ])

    elif tab == 'tab-demographics':
        return dbc.Container([
            html.H4("👥 Demographics", className="section-title"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='data-type',
                    options=[{"label": "Account", "value": "Account"}, {"label": "DFS", "value": "DFS"}],
                    value="Account", clearable=False,
                    className="custom-dropdown"
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='demographic',
                    options=[{"label": f, "value": f} for f in ["Age", "Gender", "Education", "Income"]],
                    value="Age", clearable=False,
                    className="custom-dropdown"
                ), md=6),
            ], className="mb-3"),
            dcc.Graph(id='bar-graph', **graph_container)
        ])

    elif tab == 'tab-barriers':
        return dbc.Container([
            html.H4("🚧 Barriers to Financial Inclusion", className="section-title"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='region-dropdown',
                    options=[{"label": r, "value": r} for r in sorted(df_barriers["Region"].dropna().unique())],
                    placeholder="Select Region",
                    className="custom-dropdown"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='income-dropdown',
                    options=[{"label": i, "value": i} for i in sorted(df_barriers["Income group"].dropna().unique())],
                    placeholder="Select Income Group",
                    className="custom-dropdown"
                ), md=4),
                dbc.Col(dcc.Dropdown(
                    id='country-dropdown',
                    options=[{"label": c, "value": c} for c in sorted(df_barriers["Country name"].dropna().unique())],
                    placeholder="Select Country",
                    className="custom-dropdown"
                ), md=4),
            ], className="mb-3"),
            dcc.Graph(id='barrier-graph', **graph_container)
        ])

    elif tab == 'tab-clusters':
        return dbc.Container([
            html.H4("🗺️ Clusters by Country", className="section-title"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='year_dropdown',
                    options=[{'label': y, 'value': y} for y in sorted(df['Year'].unique())],
                    value=df['Year'].max(),
                    className="custom-dropdown"
                ), md=6),
                dbc.Col(dcc.Dropdown(
                    id='region_dropdown',
                    options=[{'label': r, 'value': r} for r in sorted(df['Region'].dropna().unique())],
                    placeholder='All Regions', clearable=True,
                    className="custom-dropdown"
                ), md=6),
            ], className="mb-3"),
            dcc.Graph(id='map_graph', style={"height": "600px"}, className="graph-container")
        ])

# --- Callbacks for figures (no style changes needed) ---
@app.callback(
    Output("combined-trend", "figure"),
    Input("indicator-dropdown", "value"),
    Input("view-toggle", "value")
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

        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)

    elif view == "Rwanda":
        rwanda_df = df_filtered[(df_filtered["Indicator"] == indicator) & (df_filtered["Country name"] == "Rwanda")]
        rwanda_df = rwanda_df[["Year", "Indicator value"]]
        rwanda_df["Source"] = "Rwanda"
        trend_data.append(rwanda_df)

    combined_df = pd.concat(trend_data)
    fig = px.line(combined_df, x="Year", y="Indicator value", color="Source", markers=True,
                  title=f"{indicator} — {view} Comparison", labels={"Indicator value": "%"},
                  template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(title_font_size=20, margin=dict(l=40, r=40, t=60, b=40))
    return fig

@app.callback(
    Output("bar-graph", "figure"),
    Input("data-type", "value"), Input("demographic", "value")
)
def update_demographic_graph(data_type, demographic):
    df_selected = account_data.get(demographic) if data_type == "Account" else dfs_data.get(demographic)
    if df_selected is None or df_selected.empty:
        return px.bar(title="No data available.")
    summary = df_selected.groupby("Indicator")["Indicator value"].mean().reset_index()
    fig = px.bar(summary.sort_values(by="Indicator value", ascending=True), x="Indicator value", y="Indicator", orientation="h",
                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(l=100, r=20, t=40, b=40))
    return fig

@app.callback(
    Output("barrier-graph", "figure"),
    Input("region-dropdown", "value"),
    Input("income-dropdown", "value"),
    Input("country-dropdown", "value")
)
def update_barrier_graph(region, income, country):
    filtered = df_barriers.copy()
    if region: filtered = filtered[filtered["Region"] == region]
    if income: filtered = filtered[filtered["Income group"] == income]
    if country: filtered = filtered[filtered["Country name"] == country]
    if filtered.empty:
        return px.bar(title="No data available.")
    summary = filtered.groupby("Indicator")["Indicator value"].mean().reset_index()
    fig = px.bar(summary.sort_values(by="Indicator value", ascending=False), x="Indicator value", y="Indicator", orientation="h",
                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(l=100, r=20, t=40, b=40))
    return fig

@app.callback(
    Output("map_graph", "figure"),
    Input("year_dropdown", "value"),
    Input("region_dropdown", "value")
)
def update_map_graph(year, region):
    df_segmented = segment_countries(df, year, region)
    fig = px.choropleth(df_segmented, locations='Country name', locationmode='country names',
                        color='Segment', hover_name='Country name',
                        color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_white")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=40))
    return fig

# Run App
if __name__ == "__main__":
    app.run(debug=True)
