# Task 13: Streamlit Dashboard for Hospital CMO

**Status:** End-User Interface Task
**Dependencies:** Tasks 1-12 (uses all data sources)
**Estimated Time:** 6-8 hours
**Difficulty:** Medium

---

## Objective

Create a simple, user-friendly Streamlit dashboard that allows hospital CMOs and non-technical users to:
- Search for research papers and clinical trials
- View collaboration networks
- Explore research trends and clusters
- Generate reports
- **No technical knowledge required**

---

## Background

This is the primary interface that hospital administrators will use. It must be:
- **Simple** - No technical jargon
- **Visual** - Charts, graphs, maps
- **Fast** - Quick loading, responsive
- **Exportable** - PDF/Excel reports
- **Deployable** - Can run on Streamlit Cloud

---

## Requirements

### Pages

1. **Home/Search** - Search papers and trials by keyword, institution, author
2. **Network Explorer** - Visual collaboration network
3. **Research Trends** - Topic clusters and trends over time
4. **Geographic Map** - Where research is happening
5. **Reports** - Generate and download reports

### Features

- Simple search (like Google)
- Interactive visualizations
- Export to PDF/Excel
- Filters (date range, institution, topic)
- Mobile-friendly

---

## Architecture

```
src/dashboard/
├── __init__.py
├── app.py                    # Main Streamlit app
├── pages/
│   ├── __init__.py
│   ├── 01_🔍_Search.py       # Search page
│   ├── 02_🌐_Network.py      # Network visualization
│   ├── 03_📊_Trends.py       # Research trends
│   ├── 04_🗺️_Map.py          # Geographic map
│   └── 05_📄_Reports.py      # Report generation
├── components/
│   ├── __init__.py
│   ├── search.py             # Search component
│   ├── filters.py            # Filter sidebar
│   └── export.py             # Export utilities
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Load data from all sources
│   ├── formatting.py         # Format display
│   └── caching.py            # Streamlit caching
└── assets/
    ├── logo.png
    └── styles.css

tests/
├── __init__.py
├── test_data_loader.py
└── test_components.py
```

---

## Implementation Guide (TDD)

### Step 1: Data Loader

**Test First** (`tests/test_data_loader.py`):

```python
import pytest
from pathlib import Path
import json
from dashboard.utils.data_loader import DataLoader


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample data directory."""
    # Create unified dataset
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    unified = {
        "papers": [
            {
                "pmid": "12345",
                "title": "Cancer Research Study",
                "abstract": "This study investigates...",
                "authors": [{"name": "John Doe"}],
                "publication_date": "2024-01-15",
                "institution": "Harvard Medical School"
            }
        ],
        "trials": [
            {
                "nct_id": "NCT12345",
                "title": "Cancer Treatment Trial",
                "status": "Recruiting",
                "sponsor": "Massachusetts General Hospital",
                "conditions": ["Cancer"]
            }
        ]
    }

    with open(data_dir / "unified_dataset.json", 'w') as f:
        json.dump(unified, f)

    return tmp_path / "data"


class TestDataLoader:
    """Test data loader for dashboard."""

    def test_load_unified_data(self, sample_data_dir):
        """Test loading unified dataset."""
        loader = DataLoader(data_dir=sample_data_dir)

        data = loader.load_unified_data()

        assert "papers" in data
        assert "trials" in data
        assert len(data["papers"]) == 1
        assert len(data["trials"]) == 1

    def test_search_papers(self, sample_data_dir):
        """Test searching papers by keyword."""
        loader = DataLoader(data_dir=sample_data_dir)

        results = loader.search_papers(query="cancer")

        assert len(results) == 1
        assert results[0]["pmid"] == "12345"

    def test_search_papers_case_insensitive(self, sample_data_dir):
        """Test search is case-insensitive."""
        loader = DataLoader(data_dir=sample_data_dir)

        results = loader.search_papers(query="CANCER")

        assert len(results) == 1

    def test_filter_by_institution(self, sample_data_dir):
        """Test filtering papers by institution."""
        loader = DataLoader(data_dir=sample_data_dir)

        results = loader.filter_papers(institution="Harvard Medical School")

        assert len(results) == 1
        assert results[0]["institution"] == "Harvard Medical School"

    def test_filter_by_date_range(self, sample_data_dir):
        """Test filtering by date range."""
        loader = DataLoader(data_dir=sample_data_dir)

        results = loader.filter_papers(
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        assert len(results) == 1

    def test_get_institutions_list(self, sample_data_dir):
        """Test getting list of all institutions."""
        loader = DataLoader(data_dir=sample_data_dir)

        institutions = loader.get_institutions()

        assert "Harvard Medical School" in institutions
```

**Implementation** (`src/dashboard/utils/data_loader.py`):

```python
"""Data loader for Streamlit dashboard."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st


class DataLoader:
    """Loads and caches data for the dashboard."""

    def __init__(self, data_dir: str = "./data"):
        """Initialize data loader."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"

    @st.cache_data(ttl=3600)
    def load_unified_data(_self) -> Dict[str, Any]:
        """Load unified dataset (cached for 1 hour)."""
        # Find most recent unified dataset
        files = list(_self.processed_dir.glob("unified_dataset*.json"))

        if not files:
            # Return empty structure
            return {"papers": [], "trials": []}

        # Get most recent
        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        with open(latest_file) as f:
            data = json.load(f)

        return data

    def search_papers(
        self,
        query: str,
        fields: List[str] = ["title", "abstract"]
    ) -> List[Dict[str, Any]]:
        """Search papers by keyword."""
        data = self.load_unified_data()
        papers = data.get("papers", [])

        if not query:
            return papers

        query_lower = query.lower()
        results = []

        for paper in papers:
            # Search in specified fields
            for field in fields:
                if field in paper:
                    value = str(paper[field]).lower()
                    if query_lower in value:
                        results.append(paper)
                        break

        return results

    def filter_papers(
        self,
        institution: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter papers by various criteria."""
        data = self.load_unified_data()
        papers = data.get("papers", [])

        filtered = papers

        # Filter by institution
        if institution:
            filtered = [
                p for p in filtered
                if p.get("institution") == institution
            ]

        # Filter by date range
        if start_date:
            filtered = [
                p for p in filtered
                if p.get("publication_date", "") >= start_date
            ]

        if end_date:
            filtered = [
                p for p in filtered
                if p.get("publication_date", "") <= end_date
            ]

        return filtered

    def get_institutions(self) -> List[str]:
        """Get list of all unique institutions."""
        data = self.load_unified_data()
        papers = data.get("papers", [])

        institutions = set()
        for paper in papers:
            if "institution" in paper:
                institutions.add(paper["institution"])

        return sorted(list(institutions))

    def get_clinical_trials(
        self,
        sponsor: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get clinical trials with optional filters."""
        data = self.load_unified_data()
        trials = data.get("trials", [])

        filtered = trials

        if sponsor:
            filtered = [t for t in filtered if t.get("sponsor") == sponsor]

        if status:
            filtered = [t for t in filtered if t.get("status") == status]

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        data = self.load_unified_data()

        return {
            "total_papers": len(data.get("papers", [])),
            "total_trials": len(data.get("trials", [])),
            "institutions": len(self.get_institutions()),
            "last_updated": data.get("created_at", "Unknown")
        }
```

---

### Step 2: Main App

**Implementation** (`src/dashboard/app.py`):

```python
"""Main Streamlit dashboard application."""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Research Intelligence Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data loader
from utils.data_loader import DataLoader

@st.cache_resource
def get_data_loader():
    """Get cached data loader instance."""
    return DataLoader(data_dir="./data")

loader = get_data_loader()

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=200) if Path("assets/logo.png").exists() else None
    st.title("🔬 Research Intelligence")

    st.markdown("---")

    # Quick stats
    stats = loader.get_statistics()

    st.metric("Total Papers", f"{stats['total_papers']:,}")
    st.metric("Clinical Trials", f"{stats['total_trials']:,}")
    st.metric("Institutions", stats['institutions'])

    st.markdown("---")

    st.caption(f"Last updated: {stats['last_updated']}")

# Main content
st.markdown('<div class="main-header">Research Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
Welcome to the Research Intelligence Dashboard. This tool helps you explore:
- 📄 **Research Papers** from leading institutions
- 🧪 **Clinical Trials** from major sponsors
- 👥 **Collaboration Networks** between researchers
- 📊 **Research Trends** and emerging topics
- 🗺️ **Geographic Distribution** of research

Use the navigation on the left to explore different views.
""")

# Quick search
st.markdown("### 🔍 Quick Search")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Search papers and trials",
        placeholder="Enter keywords (e.g., 'cancer treatment', 'cardiology')",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

if query or search_button:
    if query:
        # Search papers
        papers = loader.search_papers(query)

        st.markdown(f"#### Found {len(papers)} papers")

        if papers:
            for paper in papers[:10]:  # Show top 10
                with st.expander(f"📄 {paper['title']}"):
                    st.markdown(f"**PMID:** {paper['pmid']}")
                    st.markdown(f"**Date:** {paper.get('publication_date', 'Unknown')}")
                    st.markdown(f"**Institution:** {paper.get('institution', 'Unknown')}")
                    st.markdown(f"**Abstract:** {paper.get('abstract', 'No abstract available')[:300]}...")

            if len(papers) > 10:
                st.info(f"Showing 10 of {len(papers)} results. Use filters in the Search page for more options.")
        else:
            st.warning("No results found. Try different keywords.")
    else:
        st.info("Enter a search query above.")

# Recent papers
st.markdown("### 📚 Recent Papers")

data = loader.load_unified_data()
recent_papers = sorted(
    data.get("papers", []),
    key=lambda x: x.get("publication_date", ""),
    reverse=True
)[:5]

if recent_papers:
    for paper in recent_papers:
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{paper['title']}**")
            st.caption(f"{paper.get('institution', 'Unknown')} - {paper.get('publication_date', 'Unknown')}")

        with col2:
            st.markdown(f"`PMID: {paper['pmid']}`")

# Footer
st.markdown("---")
st.caption("Research Intelligence Dashboard v1.0 | Built with Streamlit")
```

---

### Step 3: Search Page

**Implementation** (`src/dashboard/pages/01_🔍_Search.py`):

```python
"""Search page for papers and clinical trials."""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="Search", page_icon="🔍", layout="wide")

# Import data loader
from utils.data_loader import DataLoader

@st.cache_resource
def get_data_loader():
    return DataLoader(data_dir="./data")

loader = get_data_loader()

st.title("🔍 Search Papers & Clinical Trials")

# Tabs for Papers and Trials
tab1, tab2 = st.tabs(["📄 Research Papers", "🧪 Clinical Trials"])

# ===== PAPERS TAB =====
with tab1:
    st.markdown("### Search Research Papers")

    # Search box
    query = st.text_input(
        "Search by keyword",
        placeholder="e.g., cancer, cardiology, immunotherapy",
        key="papers_search"
    )

    # Filters in sidebar
    with st.sidebar:
        st.markdown("### Filters")

        institutions = ["All"] + loader.get_institutions()
        selected_institution = st.selectbox("Institution", institutions)

        # Date range
        st.markdown("**Publication Date**")
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input(
                "From",
                value=datetime.now() - timedelta(days=365),
                key="start_date"
            )

        with col2:
            end_date = st.date_input(
                "To",
                value=datetime.now(),
                key="end_date"
            )

        # Search button
        search_button = st.button("Apply Filters", type="primary", use_container_width=True)

    # Perform search
    if query or search_button:
        # Get papers
        if query:
            papers = loader.search_papers(query)
        else:
            papers = loader.load_unified_data().get("papers", [])

        # Apply filters
        if selected_institution != "All":
            papers = [p for p in papers if p.get("institution") == selected_institution]

        # Date filter
        papers = [
            p for p in papers
            if start_date.isoformat() <= p.get("publication_date", "") <= end_date.isoformat()
        ]

        # Display results
        st.markdown(f"### Results: {len(papers)} papers found")

        if papers:
            # Convert to dataframe for display
            df = pd.DataFrame([
                {
                    "PMID": p["pmid"],
                    "Title": p["title"],
                    "Institution": p.get("institution", "Unknown"),
                    "Date": p.get("publication_date", "Unknown")
                }
                for p in papers
            ])

            # Display as interactive table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # Detailed view
            st.markdown("### Detailed View")

            for i, paper in enumerate(papers):
                with st.expander(f"{i+1}. {paper['title']}", expanded=(i == 0)):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**PMID:** {paper['pmid']}")

                    with col2:
                        st.markdown(f"**Date:** {paper.get('publication_date', 'Unknown')}")

                    with col3:
                        st.markdown(f"**Institution:** {paper.get('institution', 'Unknown')}")

                    st.markdown("**Abstract:**")
                    st.write(paper.get("abstract", "No abstract available."))

                    if "authors" in paper:
                        st.markdown("**Authors:**")
                        authors = ", ".join([a.get("name", "Unknown") for a in paper["authors"]])
                        st.write(authors)

            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"papers_search_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        else:
            st.warning("No papers found matching your criteria.")

# ===== TRIALS TAB =====
with tab2:
    st.markdown("### Search Clinical Trials")

    # Search/filter
    sponsor_filter = st.text_input(
        "Filter by sponsor",
        placeholder="e.g., Massachusetts General Hospital",
        key="trials_sponsor"
    )

    status_options = ["All", "Recruiting", "Active", "Completed", "Terminated"]
    status_filter = st.selectbox("Trial Status", status_options)

    # Get trials
    trials = loader.get_clinical_trials()

    # Apply filters
    if sponsor_filter:
        trials = [t for t in trials if sponsor_filter.lower() in t.get("sponsor", "").lower()]

    if status_filter != "All":
        trials = [t for t in trials if t.get("status") == status_filter]

    # Display
    st.markdown(f"### Results: {len(trials)} trials found")

    if trials:
        for trial in trials:
            with st.expander(f"🧪 {trial['title']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**NCT ID:** {trial['nct_id']}")
                    st.markdown(f"**Status:** {trial.get('status', 'Unknown')}")

                with col2:
                    st.markdown(f"**Sponsor:** {trial.get('sponsor', 'Unknown')}")
                    st.markdown(f"**Conditions:** {', '.join(trial.get('conditions', []))}")
    else:
        st.info("No trials found.")
```

---

### Step 4: Network Page

**Implementation** (`src/dashboard/pages/02_🌐_Network.py`):

```python
"""Network visualization page."""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from utils.data_loader import DataLoader

st.set_page_config(page_title="Network", page_icon="🌐", layout="wide")

st.title("🌐 Collaboration Network")

st.markdown("""
Explore the collaboration network between researchers and institutions.
""")

# Load data
@st.cache_resource
def load_network_data():
    """Load network data from Neo4j or files."""
    # This would connect to Neo4j or load network JSON
    # For now, create a sample network
    G = nx.Graph()

    G.add_edge("Harvard", "MIT", weight=10)
    G.add_edge("Harvard", "Stanford", weight=5)
    G.add_edge("MIT", "Stanford", weight=8)
    G.add_edge("MIT", "Berkeley", weight=6)

    return G

G = load_network_data()

# Visualization options
with st.sidebar:
    st.markdown("### Network Options")

    layout = st.selectbox(
        "Layout",
        ["Spring", "Circular", "Kamada-Kawai"]
    )

    show_labels = st.checkbox("Show Labels", value=True)

# Create network visualization
if layout == "Spring":
    pos = nx.spring_layout(G)
elif layout == "Circular":
    pos = nx.circular_layout(G)
else:
    pos = nx.kamada_kawai_layout(G)

# Create Plotly figure
edge_trace = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]

    edge_trace.append(
        go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        )
    )

node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    mode='markers+text' if show_labels else 'markers',
    text=[node for node in G.nodes()] if show_labels else None,
    textposition="top center",
    marker=dict(
        size=20,
        color='#1f77b4',
        line=dict(width=2, color='white')
    ),
    hovertext=[f"{node}<br>Connections: {G.degree(node)}" for node in G.nodes()],
    hoverinfo='text'
)

fig = go.Figure(data=edge_trace + [node_trace])

fig.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=0, l=0, r=0, t=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Network statistics
st.markdown("### Network Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Nodes", G.number_of_nodes())

with col2:
    st.metric("Total Edges", G.number_of_edges())

with col3:
    density = nx.density(G)
    st.metric("Network Density", f"{density:.2%}")

with col4:
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    st.metric("Avg Connections", f"{avg_degree:.1f}")
```

---

## Running the Dashboard

```bash
# Install dependencies
pip install streamlit plotly pandas networkx

# Run dashboard
streamlit run src/dashboard/app.py
```

---

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select `src/dashboard/app.py` as main file
5. Deploy!

---

## Success Criteria

- [ ] Dashboard runs locally
- [ ] All pages functional
- [ ] Search works correctly
- [ ] Visualizations render
- [ ] Export features work
- [ ] Mobile-friendly
- [ ] Deployable to Streamlit Cloud

---

**End of Task 13 Specification**
