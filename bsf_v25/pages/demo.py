import streamlit as st
import time
from utils.styling import *
from utils.engine import run_full_pipeline, SEGMENT_COLORS as SEG_C, RISK_COLORS as RC

PRESETS = {
    "Ahmed Al-Harbi (VIP)": {
        'name': 'Ahmed Al-Harbi', 'monthly_income': 32000, 'account_balance': 280000,
        'credit_score': 780, 'tenure_months': 96, 'late_payment_count': 0,
        'total_spend_12m': 174000, 'app_logins_monthly': 22, 'txn_frequency_12m': 95,
        'products_owned': 'SA_SAVING', 'preferred_channel': 'Mobile App', 'employment': 'Government',
    },
    "Sara Al-Otaibi (Gold)": {
        'name': 'Sara Al-Otaibi', 'monthly_income': 14000, 'account_balance': 65000,
        'credit_score': 710, 'tenure_months': 48, 'late_payment_count': 1,
        'total_spend_12m': 72000, 'app_logins_monthly': 12, 'txn_frequency_12m': 55,
        'products_owned': 'SA_SAVING|CC_CASHBACK', 'preferred_channel': 'Mobile App', 'employment': 'Private',
    },
    "Khalid Al-Dosari (Standard)": {
        'name': 'Khalid Al-Dosari', 'monthly_income': 4500, 'account_balance': 8000,
        'credit_score': 590, 'tenure_months': 14, 'late_payment_count': 3,
        'total_spend_12m': 28000, 'app_logins_monthly': 3, 'txn_frequency_12m': 22,
        'products_owned': 'SA_SAVING', 'preferred_channel': 'Branch', 'employment': 'Private',
    },
    "Custom Profile": None,
}

def render():
    st.markdown(page_header("Live Customer Demo",
        "Enter customer data to run the full BSF AI pipeline in real-time", "🎯"), unsafe_allow_html=True)

    # ── Preset selector ───────────────────────────────────────
    preset = st.selectbox("⚡ Quick Load Preset:", list(PRESETS.keys()))
    data   = PRESETS.get(preset) or {}

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────
    with st.form("customer_form"):
        st.markdown(f'<p style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:16px;">👤 Customer Profile</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            name      = st.text_input("Full Name",              value=data.get('name', 'New Customer'))
            income    = st.number_input("Monthly Income (SAR)", value=int(data.get('monthly_income', 10000)), min_value=0, step=500)
            balance   = st.number_input("Account Balance (SAR)",value=int(data.get('account_balance', 50000)), min_value=0, step=1000)
            score     = st.number_input("Credit Score (300–900)",value=int(data.get('credit_score', 680)), min_value=300, max_value=900, step=10)
        with c2:
            tenure    = st.number_input("Tenure (months)",      value=int(data.get('tenure_months', 24)), min_value=0, step=6)
            late      = st.number_input("Late Payments",         value=int(data.get('late_payment_count', 0)), min_value=0, max_value=20)
            spend     = st.number_input("Annual Spend (SAR)",    value=int(data.get('total_spend_12m', 60000)), min_value=0, step=5000)
            logins    = st.number_input("App Logins/Month",      value=int(data.get('app_logins_monthly', 8)), min_value=0)
        with c3:
            products  = st.text_input("Products Owned (pipe-separated)",
                                       value=data.get('products_owned', 'SA_SAVING'),
                                       help="e.g. SA_SAVING|CC_CASHBACK")
            channel   = st.selectbox("Preferred Channel",
                                      ['Mobile App', 'Branch', 'WhatsApp', 'ATM', 'Online Banking'],
                                      index=['Mobile App','Branch','WhatsApp','ATM','Online Banking'].index(
                                          data.get('preferred_channel','Mobile App')))
            employment= st.selectbox("Employment",
                                      ['Government', 'Private', 'Self-Employed', 'Retired', 'Student'],
                                      index=['Government','Private','Self-Employed','Retired','Student'].index(
                                          data.get('employment','Private')))
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Run Full AI Pipeline", use_container_width=True)

    # ── Run Pipeline ──────────────────────────────────────────
    if submitted:
        with st.spinner("⏳ Running BSF AI Pipeline..."):
            progress = st.progress(0)
            steps_labels = [
                "Loading customer data...",
                "Running segmentation model...",
                "Computing NBA recommendations...",
                "Analyzing behavior patterns...",
                "Generating credit decision...",
                "Preparing results...",
            ]
            bar = st.empty()
            for i, lbl in enumerate(steps_labels):
                bar.info(f"Step {i+1}/6: {lbl}")
                progress.progress((i+1)/6)
                time.sleep(0.3)
            bar.empty()
            progress.empty()

            customer_input = {
                'monthly_income': income, 'account_balance': balance,
                'credit_score': score, 'tenure_months': tenure,
                'late_payment_count': late, 'total_spend_12m': spend,
                'app_logins_monthly': logins, 'products_owned': products,
                'preferred_channel': channel,
            }
            result = run_full_pipeline(customer_input)

        st.success("✅ Pipeline complete — results below")
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Step Indicator ────────────────────────────────────
        steps_done = ["Data Input", "Segmentation", "NBA Engine", "Behavior", "Credit Decision", "Results Ready"]
        step_html  = " › ".join([
            f'<span style="color:{ABG_GREEN};font-weight:700;font-size:12px;">✓ {s}</span>'
            if i < 5 else
            f'<span style="color:{ABG_BLUE};font-weight:700;font-size:12px;">● {s}</span>'
            for i, s in enumerate(steps_done)
        ])
        st.markdown(f'<div style="padding:12px;background:{ABG_WHITE};border:1px solid {ABG_BORDER};border-radius:8px;margin-bottom:20px;">{step_html}</div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # MODULE 1 RESULTS
        # ══════════════════════════════════════════════════════
        st.markdown(f'<h3 style="color:{ABG_DARK};font-size:15px;font-weight:800;margin:0 0 12px;">👥 Module 1 — Segmentation & NBA Results</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])

        with c1:
            seg = result['segment']
            eng = result['engagement_score']
            ch  = result['channel']
            seg_color = SEG_C.get(seg, ABG_MUTED)

            _seg_pill  = segment_pill(seg)
            _prog_bar  = progress_bar(min(100, eng / 30 * 100), seg_color)
            _ch_badge  = channel_badge(ch['primary'])
            st.markdown(
                f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
                f'border-radius:12px;padding:22px;border-top:3px solid {seg_color};">'
                f'<div style="font-size:12px;color:{ABG_MUTED};margin-bottom:8px;">CUSTOMER SEGMENT</div>'
                f'{_seg_pill}'
                f'<div style="font-size:12px;color:{ABG_MUTED};margin-top:18px;margin-bottom:4px;">ENGAGEMENT SCORE</div>'
                f'<div style="font-size:22px;font-weight:800;color:{seg_color};">{eng:.1f} / 100</div>'
                f'{_prog_bar}'
                f'<div style="font-size:12px;color:{ABG_MUTED};margin-top:16px;margin-bottom:6px;">CONTACT CHANNEL</div>'
                f'{_ch_badge}'
                f'<div style="font-size:11px;color:{ABG_MUTED};margin-top:6px;">Backup: {ch["backup"]} · <em>{ch["reason"]}</em></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            nba = result['nba_recommendations']
            if nba:
                st.markdown(f'<div style="font-size:12px;color:{ABG_MUTED};margin-bottom:8px;">NEXT-BEST ACTION RECOMMENDATIONS</div>', unsafe_allow_html=True)
                for i, rec in enumerate(nba, 1):
                    st.markdown(nba_card(rec, i), unsafe_allow_html=True)
            else:
                st.markdown(info_box("No eligible products found for this customer profile.", ABG_MUTED), unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # MODULE 2 RESULTS
        # ══════════════════════════════════════════════════════
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<h3 style="color:{ABG_DARK};font-size:15px;font-weight:800;margin:0 0 12px;">📊 Module 2 — Behavior Analysis Results</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            beh_score = result['behavior_score']
            beh_cat   = result['behavior_category']
            beh_color = {
                'Highly Active': ABG_GREEN, 'Active': ABG_BLUE,
                'Moderate': ABG_GOLD, 'Dormant': ABG_MUTED
            }.get(beh_cat, ABG_MUTED)
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;border-top:3px solid {beh_color};">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:6px;">BEHAVIOR SCORE</div>
              <div style="font-size:32px;font-weight:800;color:{beh_color};">{beh_score} <span style="font-size:14px;font-weight:400;color:{ABG_MUTED};">/ 100</span></div>
              {progress_bar(beh_score, beh_color, '10px')}
              <span style="display:inline-block;margin-top:10px;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;background:{beh_color}15;color:{beh_color};">{beh_cat}</span>
            </div>""", unsafe_allow_html=True)

        with c2:
            anom  = result['anomaly']
            drift = result['drift_status']
            drift_colors = {'Upgrade Candidate': ABG_GREEN, 'Stable': ABG_BLUE, 'Downgrade Risk': ABG_RED}
            drift_icons  = {'Upgrade Candidate': '⬆️', 'Stable': '➡️', 'Downgrade Risk': '⬇️'}
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;border-top:3px solid {'#E63946' if anom['flag'] else ABG_GREEN};">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:8px;">ANOMALY STATUS</div>
              <div style="font-size:18px;font-weight:700;color:{'#E63946' if anom['flag'] else ABG_GREEN};">
                {'⚠️ Anomaly Detected' if anom['flag'] else '✅ Normal Pattern'}
              </div>
              {''.join([f"<div style='font-size:11px;color:{ABG_MUTED};margin-top:4px;'>• {r}</div>" for r in anom['reasons']]) if anom['flag'] else ''}

              <div style="font-size:12px;color:{ABG_MUTED};margin-top:14px;margin-bottom:6px;">SEGMENT DRIFT</div>
              <div style="font-size:16px;font-weight:700;color:{drift_colors.get(drift, ABG_MUTED)};">
                {drift_icons.get(drift, '→')} {drift}
              </div>
            </div>""", unsafe_allow_html=True)

        with c3:
            insights = result['agent_insights']
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;border-top:3px solid {ABG_BLUE};">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:10px;">ANALYSIS AGENT INSIGHTS</div>
              {''.join([f"<div style='font-size:12px;color:{ABG_DARK};line-height:1.6;padding:4px 0;border-bottom:1px solid {ABG_BORDER};'>{ins}</div>" for ins in insights])}
            </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # MODULE 3 RESULTS
        # ══════════════════════════════════════════════════════
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<h3 style="color:{ABG_DARK};font-size:15px;font-weight:800;margin:0 0 12px;">💳 Module 3 — Credit Decision Results</h3>', unsafe_allow_html=True)

        risk_cat   = result['risk_category']
        risk_score = result['risk_score']
        credit     = result['credit']
        approval   = result['approval']
        risk_color = RC.get(risk_cat, ABG_MUTED)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;border-top:3px solid {risk_color};">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:8px;">RISK ASSESSMENT</div>
              {risk_pill(risk_cat)}
              <div style="font-size:12px;color:{ABG_MUTED};margin-top:12px;margin-bottom:4px;">Risk Score</div>
              <div style="font-size:24px;font-weight:800;color:{risk_color};">{risk_score} <span style="font-size:12px;color:{ABG_MUTED};">/ 100</span></div>
              {progress_bar(risk_score, risk_color, '8px')}
            </div>""", unsafe_allow_html=True)

        with c2:
            cc_val = f"SAR {credit['cc_limit']:,}" if credit['cc_eligible'] else "Not Eligible"
            cc_col = ABG_BLUE if credit['cc_eligible'] else ABG_MUTED
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;border-top:3px solid {ABG_BLUE};">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:6px;">💳 CREDIT CARD</div>
              <div style="font-size:22px;font-weight:800;color:{cc_col};">{cc_val}</div>
              <div style="font-size:11px;color:{ABG_MUTED};margin-top:4px;">Tier: {credit['cc_card_tier']} · APR: {credit['cc_apr']}%</div>

              <div style="font-size:12px;color:{ABG_MUTED};margin-top:16px;margin-bottom:6px;">🏦 PERSONAL FINANCE</div>
              <div style="font-size:22px;font-weight:800;color:{ABG_GOLD if credit['loan_eligible'] else ABG_MUTED};">
                {'SAR ' + f"{credit['loan_amount']:,}" if credit['loan_eligible'] else 'Not Eligible'}
              </div>
              <div style="font-size:11px;color:{ABG_MUTED};margin-top:4px;">
                {'Rate: ' + str(credit['interest_rate']) + '% p.a. · ' + str(credit['loan_tenure_months']) + 'm · SAR ' + f"{credit['loan_installment']:,}" + '/month' if credit['loan_eligible'] else ''}
              </div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:12px;padding:20px;">
              <div style="font-size:12px;color:{ABG_MUTED};margin-bottom:8px;">APPROVAL WORKFLOW</div>
              {approval_box(approval)}
            </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # CHATBOT PREVIEW (Module 4)
        # ══════════════════════════════════════════════════════
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<h3 style="color:{ABG_DARK};font-size:15px;font-weight:800;margin:0 0 12px;">💬 Module 4 Preview — BSF Chatbot Welcome Message</h3>', unsafe_allow_html=True)

        nba = result['nba_recommendations']
        first_name = name.split()[0] if name else 'Customer'
        prod1 = nba[0]['product_name'] if nba else 'BSF Products'
        prod2 = nba[1]['product_name'] if len(nba) > 1 else None
        cat1  = nba[0]['category'] if nba else ''
        cat2  = nba[1]['category'] if len(nba) > 1 else ''
        score1= nba[0]['match_score'] if nba else 0
        reason1 = nba[0]['reason_codes'][0] if nba and nba[0]['reason_codes'] else 'your profile'
        reason2 = nba[1]['reason_codes'][0] if len(nba) > 1 and nba[1]['reason_codes'] else 'your eligibility profile'
        icon1 = '💳' if 'Credit Card' in cat1 else '💰'
        icon2 = '💳' if 'Credit Card' in cat2 else '💰'

        welcome = f"""Hello <strong>{first_name}</strong> 👋<br><br>
Welcome back to BSF Digital Banking.<br><br>
Based on your recent banking activity and financial profile, we have identified products that may be relevant to your needs:<br><br>
{icon1} <strong>{prod1}</strong> — Match Score <strong>{score1}%</strong><br>
<span style="color:{ABG_MUTED};font-size:13px;">Recommended because of your {reason1.lower()}.</span><br><br>
{f'{icon2} <strong>{prod2}</strong><br><span style="color:{ABG_MUTED};font-size:13px;">Recommended based on your {reason2.lower()}.</span><br><br>' if prod2 else ''}
Would you like to know why these products were recommended, compare their benefits, or explore other products available to you?"""

        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_GOLD}33;border-radius:12px;padding:24px;
        border-left:4px solid {ABG_GOLD};">
          <div style="font-size:14px;line-height:1.9;color:{ABG_DARK};">{welcome}</div>
        </div>""", unsafe_allow_html=True)

        # ── Full Summary ──────────────────────────────────────
        with st.expander("📋 Full Pipeline Summary"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Segment:** {result['segment']}")
                st.markdown(f"**Engagement Score:** {result['engagement_score']:.1f}")
                st.markdown(f"**Primary Channel:** {result['channel']['primary']}")
                st.markdown(f"**Behavior Score:** {result['behavior_score']}")
                st.markdown(f"**Behavior Category:** {result['behavior_category']}")
            with c2:
                st.markdown(f"**Risk Category:** {result['risk_category']}")
                st.markdown(f"**Risk Score:** {result['risk_score']}")
                cc_limit_str = f"SAR {result['credit']['cc_limit']:,}" if result['credit']['cc_eligible'] else 'Not Eligible'
                loan_amt_str = f"SAR {result['credit']['loan_amount']:,}" if result['credit']['loan_eligible'] else 'Not Eligible'
                st.markdown(f"**CC Limit:** {cc_limit_str}")
                st.markdown(f"**Loan Amount:** {loan_amt_str}")
                st.markdown(f"**Approval:** {result['approval']['status']}")
