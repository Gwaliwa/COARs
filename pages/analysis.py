# pages/Analysis.py
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="COAR Analysis", layout="wide")
st.title("📊 COAR: Counts per Region & Year + Keyword Windows")

# ---------------- Load consolidated data from session ----------------
if "consolidated_df" not in st.session_state:
    st.warning("No consolidated data found. Go to Home and click 'Consolidate by Headings' first.")
    st.stop()

base_df = st.session_state["consolidated_df"].copy()
st.success(f"Loaded consolidated table with {len(base_df)} rows.")
st.dataframe(base_df.head(), use_container_width=True)

# ---------------- Session: categories for pattern counts ----------------
if "categories" not in st.session_state:
    st.session_state.categories = {}   # { "CategoryName": "regex|pattern|..." }

def add_or_merge_category(cat: str, pattern: str):
    cat = cat.strip()
    if not cat or not pattern.strip():
        return
    st.session_state.categories[cat] = (
        f"{st.session_state.categories.get(cat, '')}|{pattern}"
        if cat in st.session_state.categories else pattern
    )

# ---------------- Filters (shared by both tools) ----------------
with st.expander("Filters (optional)"):
    base_df["year"] = pd.to_numeric(base_df["year"], errors="coerce").astype("Int64")
    all_years = sorted([int(y) for y in base_df["year"].dropna().unique()])
    all_regions = sorted(base_df["unicef_region"].dropna().unique())

    sel_years = st.multiselect("Filter by year", all_years, default=all_years)
    sel_regions = st.multiselect("Filter by UNICEF region", all_regions, default=all_regions)

    filtered_df = base_df.copy()
    if sel_years:
        filtered_df = filtered_df[filtered_df["year"].isin(sel_years)]
    if sel_regions:
        filtered_df = filtered_df[filtered_df["unicef_region"].isin(sel_regions)]

    st.caption(f"Filtered rows: {len(filtered_df)}")

protected_cols = {"filename", "filepath", "country", "year", "unicef_region"}
heading_candidates = [c for c in filtered_df.columns if c not in protected_cols]

# ---------------- Tabs ----------------
tab_counts, tab_windows = st.tabs(["🧮 Pattern counts (Region & Year)", "🧷 Keyword windows (context sentences)"])

# ======================================================================================
# TAB 1: Pattern counts (Region & Year)
# ======================================================================================
with tab_counts:
    st.subheader("🧮 Choose headings and patterns for counting")

    headings_chosen = st.multiselect(
        "Text columns to search (headings)",
        heading_candidates,
        default=heading_candidates[:1] if heading_candidates else []
    )
    if not headings_chosen:
        st.info("Select at least one heading column to analyze.")
    else:
        with st.expander("Add categories and patterns", expanded=True):
            left, right = st.columns([3,2])

            with left:
                mode = st.radio("Add patterns via…", ["Manual", "Excel"], horizontal=True)
                if mode == "Excel":
                    st.write("Excel schemas accepted: (Category, Pattern) **or** (Category, Term)")
                    xfile = st.file_uploader("Upload Excel", type=["xlsx","xls"], key="cats_excel_upl")
                    if xfile is not None:
                        try:
                            xdf = pd.read_excel(xfile)
                            cols = {c.lower(): c for c in xdf.columns}
                            if "category" not in cols:
                                st.error("Excel must include a 'Category' column.")
                            else:
                                loaded = {}
                                if "pattern" in cols:
                                    for _, r in xdf.iterrows():
                                        cat = str(r[cols["category"]]).strip()
                                        pat = str(r[cols["pattern"]]).strip()
                                        if cat and pat:
                                            loaded[cat] = (f"{loaded.get(cat,'')}|{pat}") if cat in loaded else pat
                                elif "term" in cols:
                                    grp = xdf.groupby(cols["category"])[cols["term"]].apply(list)
                                    for cat, terms in grp.items():
                                        terms = [str(t).strip() for t in terms if str(t).strip()]
                                        loaded[cat] = "|".join(re.escape(t) for t in terms)
                                else:
                                    st.error("Need either ('Category','Pattern') or ('Category','Term') columns.")
                                    loaded = {}

                                for k, v in loaded.items():
                                    add_or_merge_category(k, v)
                                if loaded:
                                    st.success(f"Loaded {len(loaded)} categories.")
                        except Exception as e:
                            st.error(f"Failed to read Excel: {e}")

                else:
                    with st.form("manual_add"):
                        c1, c2 = st.columns(2)
                        with c1:
                            cat_name = st.text_input("Category name", value="oos")
                            as_regex = st.checkbox("Treat input as full regex", value=True)
                        with c2:
                            word_boundaries = st.checkbox("Wrap literals with \\b (only when NOT regex)", value=False)
                        patterns_text = st.text_area(
                            "Terms/patterns (comma or newline separated).",
                            "out-of-sch\nout of school\nschool dropout\nschool retent",
                            height=120
                        )
                        go = st.form_submit_button("Add category")

                    if go:
                        parts = [p.strip() for chunk in patterns_text.split("\n") for p in chunk.split(",")]
                        parts = [p for p in parts if p]
                        if not parts:
                            st.warning("No terms provided.")
                        else:
                            if as_regex:
                                patt = "|".join(parts)
                            else:
                                escaped = [re.escape(x) for x in parts]
                                if word_boundaries:
                                    escaped = [rf"\b{e}\b" for e in escaped]
                                patt = "|".join(escaped)
                            add_or_merge_category(cat_name, patt)
                            st.success(f"Added/merged category '{cat_name}'.")

            with right:
                st.markdown("#### Categories in session")
                if st.session_state.categories:
                    st.dataframe(
                        pd.DataFrame(
                            [{"Category": k, "Pattern (regex)": v} for k, v in st.session_state.categories.items()]
                        ),
                        use_container_width=True
                    )
                    cats_df = pd.DataFrame(
                        [{"Category": k, "Pattern": v} for k, v in st.session_state.categories.items()]
                    )
                    st.download_button(
                        "💾 Download categories.csv",
                        data=cats_df.to_csv(index=False),
                        file_name="categories.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    if st.button("🧹 Clear categories"):
                        st.session_state.categories = {}
                        st.rerun()
                else:
                    st.info("No categories added yet.")

        # Quick-load robust OOS example
        with st.expander("Quick-load example: OOS pattern"):
            if st.button("Load robust OOS"):
                st.session_state.categories["oos"] = (
                    r"\bout\s*(?:-| )*of\s*(?:-| )*sch(?:ool)?\b|"
                    r"\bschool\s*-?\s*drop(?:out)?\b|"
                    r"\bschool\s*-?\s*retent\w*\b"
                )
                st.success("Loaded OOS pattern.")

        st.divider()

        # ---------------- Count & Aggregate: Region + Year ----------------
        if st.button("🔎 Count patterns and aggregate by Region & Year", use_container_width=True):
            if not st.session_state.categories:
                st.warning("Please add at least one category first.")
                st.stop()

            data = filtered_df.copy()
            for h in headings_chosen:
                data[h] = data[h].astype(str)

            # Count matches into columns like <heading>_<category>
            for cat, pattern in st.session_state.categories.items():
                for h in headings_chosen:
                    out_col = f"{h}_{cat}"
                    data[out_col] = data[h].str.count(pattern, flags=re.IGNORECASE)

            # Build long format for aggregation
            count_cols = [c for c in data.columns if any(c.startswith(h + "_") for h in headings_chosen)]
            if not count_cols:
                st.info("No counts produced. Check your patterns/headings.")
            else:
                long = data.melt(
                    id_vars=["filename", "country", "unicef_region", "year"],
                    value_vars=count_cols,
                    var_name="heading_category",
                    value_name="count"
                )
                split = long["heading_category"].str.rsplit("_", n=1, expand=True)
                long["heading"] = split[0]
                long["category"] = split[1]
                long = long.drop(columns=["heading_category"])

                # ---- Aggregations ----
                st.subheader("📍 Totals by UNICEF Region")
                by_region = long.groupby(["unicef_region", "category"], as_index=False)["count"].sum()
                pivot_region = by_region.pivot(index="unicef_region", columns="category", values="count").fillna(0).astype(int)
                st.dataframe(pivot_region, use_container_width=True)

                st.subheader("📅 Totals by Year")
                by_year = long.groupby(["year", "category"], as_index=False)["count"].sum()
                pivot_year = by_year.pivot(index="year", columns="category", values="count").fillna(0).astype(int)
                st.dataframe(pivot_year, use_container_width=True)

                st.subheader("🗺️ Region × Year (all categories combined)")
                ry = long.groupby(["unicef_region", "year"], as_index=False)["count"].sum()
                pivot_ry = ry.pivot(index="unicef_region", columns="year", values="count").fillna(0).astype(int)
                st.dataframe(pivot_ry, use_container_width=True)

                # Quick charts (totals across categories)
                st.subheader("📊 Bar charts (totals across categories)")

                # Regions
                fig1 = plt.figure()
                region_totals = by_region.groupby("unicef_region")["count"].sum().sort_values(ascending=False)
                plt.bar(region_totals.index, region_totals.values)
                plt.xlabel("UNICEF Region"); plt.ylabel("Total Matches"); plt.title("Totals by Region")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig1)

                # Years
                fig2 = plt.figure()
                year_totals = by_year.groupby("year")["count"].sum().sort_values()
                x_years = [int(y) for y in year_totals.index.tolist() if pd.notna(y)]
                plt.bar(x_years, year_totals.values)
                plt.xlabel("Year"); plt.ylabel("Total Matches"); plt.title("Totals by Year")
                plt.xticks(rotation=0)
                st.pyplot(fig2)

                # Downloads
                st.subheader("💾 Download Aggregates")
                st.download_button(
                    "Download by Region (CSV)",
                    pivot_region.to_csv(),
                    "counts_by_region.csv", "text/csv",
                    use_container_width=True
                )
                st.download_button(
                    "Download by Year (CSV)",
                    pivot_year.to_csv(),
                    "counts_by_year.csv", "text/csv",
                    use_container_width=True
                )
                st.download_button(
                    "Download Region×Year (CSV)",
                    pivot_ry.to_csv(),
                    "counts_region_year.csv", "text/csv",
                    use_container_width=True
                )

# ======================================================================================
# TAB 2: Keyword windows (context sentences)
# ======================================================================================
with tab_windows:
    st.subheader("🔍 Extract Keyword Windows from Text Columns (user-provided keywords)")

    # Inputs
    keyword_input = st.text_area(
        "Enter keywords or phrases (comma or newline separated):",
        "out-of-sch\nout of school\nschool dropout\ninclusive education",
        height=120
    )
    N = st.number_input("Sentences before/after to include", min_value=0, max_value=5, value=1)

    # Offer only object/text-like columns, excluding protected
    text_like_cols = [c for c in filtered_df.columns if c not in protected_cols and filtered_df[c].dtype == object]
    text_columns = st.multiselect(
        "Select text columns to search:",
        text_like_cols,
        default=text_like_cols[:1] if text_like_cols else []
    )

    # Build pattern
    keywords = [k.strip() for chunk in keyword_input.split("\n") for k in chunk.split(",")]
    keywords = [k for k in keywords if k]
    if not keywords:
        st.info("Add at least one keyword to proceed.")
    else:
        joined_pattern = re.compile("|".join(map(re.escape, keywords)), flags=re.IGNORECASE)
        SENT_SPLIT_REGEX = r'(?<=[\.\?\!])\s+'

        def _join_clean(parts):
            return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())

        def extract_windows_for_series(s: pd.Series, pattern: re.Pattern, N: int = 1) -> pd.Series:
            s = s.fillna("").astype(str)
            exploded = s.str.split(SENT_SPLIT_REGEX, regex=True).explode()
            hit = exploded.str.contains(pattern, na=False)
            win = (hit.astype(int)
                     .groupby(level=0)
                     .rolling(window=2*N+1, center=True, min_periods=1)
                     .max()
                     .reset_index(level=1, drop=True)
                     .astype(bool))
            kept = exploded[win]
            out = kept.groupby(level=0).agg(_join_clean)
            return out.reindex(s.index).fillna("")

        if st.button("Extract Keyword Windows", use_container_width=True, key="extract_kw_windows"):
            df_windows = filtered_df.copy()
            created_cols = []
            for col in text_columns:
                new_col = f"{col}_extracted_relevant_text"
                df_windows[new_col] = extract_windows_for_series(df_windows[col], joined_pattern, N)
                created_cols.append(new_col)

            if created_cols:
                st.success(f"Extraction complete! Created {len(created_cols)} column(s).")
                st.dataframe(df_windows[created_cols].head(50), use_container_width=True)

                # ---------------- Wide download (one row per file, one column per heading) ----------------
                id_cols = [c for c in ["filename", "country", "unicef_region", "year"] if c in df_windows.columns]
                export_cols = id_cols + created_cols
                st.download_button(
                    "💾 Download Extracted Windows (WIDE CSV)",
                    df_windows[export_cols].to_csv(index=False),
                    file_name="keyword_windows_wide.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # ---------------- NEW: Full dataset with extracted columns ----------------
                st.download_button(
                    "💾 Download FULL dataset + extracted columns (CSV)",
                    df_windows.to_csv(index=False),
                    file_name="keyword_windows_full_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # ---------------- NEW: Extracted sentences (LONG CSV, one row per sentence) --------------
                st.subheader("🧾 Extracted sentences (one row per sentence)")
                all_cols = df_windows.columns.tolist()
                sentence_rows = []
                for idx, r in df_windows.iterrows():
                    base_meta = r.to_dict()  # includes ALL previous columns
                    for col in created_cols:
                        txt = str(r[col]) if pd.notna(r[col]) else ""
                        if not txt.strip():
                            continue
                        # split into sentences
                        for sent in re.split(SENT_SPLIT_REGEX, txt):
                            sent = sent.strip()
                            if not sent:
                                continue
                            row = dict(base_meta)  # copy all previous columns
                            row["heading"] = col.replace("_extracted_relevant_text", "")
                            row["extracted_sentence"] = sent
                            sentence_rows.append(row)

                if sentence_rows:
                    sentences_df = pd.DataFrame(sentence_rows)
                    st.dataframe(sentences_df.head(100), use_container_width=True)
                    st.download_button(
                        "💾 Download Extracted Sentences (LONG CSV)",
                        sentences_df.to_csv(index=False),
                        file_name="keyword_windows_sentences_long.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No sentences to export. Try widening the window or adding more keywords.")

                # ---------------- Long/tidy contracted results -------------------------------------------
                st.subheader("🧱 Contracted results (long/tidy)")
                group_per_heading = st.checkbox(
                    "Make one row per file AND heading",
                    value=True,
                    help="If OFF, one row per file (all selected headings concatenated)."
                )

                # Build long dataframe: id cols + heading + contracted_text
                long_parts = []
                for col in created_cols:
                    heading_name = col.replace("_extracted_relevant_text", "")
                    part = df_windows[id_cols + [col]].rename(columns={col: "contracted_text"})
                    part["heading"] = heading_name
                    long_parts.append(part)
                long_df = pd.concat(long_parts, ignore_index=True)
                long_df["contracted_text"] = long_df["contracted_text"].fillna("").astype(str).str.strip()
                long_df = long_df[long_df["contracted_text"] != ""]

                if not group_per_heading:
                    agg = long_df.groupby(id_cols, dropna=False, as_index=False)["contracted_text"] \
                                 .apply(lambda s: " || ".join(s))
                    st.dataframe(agg, use_container_width=True)
                    st.download_button(
                        "💾 Download Contracted (one row per file)",
                        agg.to_csv(index=False),
                        file_name="keyword_windows_contracted_per_file.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    tidy = long_df[id_cols + ["heading", "contracted_text"]]
                    st.dataframe(tidy, use_container_width=True)
                    st.download_button(
                        "💾 Download Contracted (one row per file+heading)",
                        tidy.to_csv(index=False),
                        file_name="keyword_windows_contracted_per_file_heading.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No text columns selected or nothing was created.")
