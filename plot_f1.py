import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_figure_1():
    # Setup the figure and axes for a clean, paper-like layout
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off') # Turn off standard axes

    # Typography settings to mimic LaTeX
    plt.rcParams.update({
        "text.usetex": False, # Set to True if you have a full LaTeX installation for perfect rendering
        "font.family": "serif",
        "font.size": 10,
    })
    title_font = {'fontname':'sans-serif', 'weight':'bold', 'size':14}
    header_font = {'fontname':'sans-serif', 'weight':'bold', 'size':12}
    example_font = {'family':'monospace', 'size': 9}
    math_font = {'style':'italic'}

    # Defines color scheme
    prev_work_bg = "#f0f4f8" # Light gray-blue
    prev_work_border = "#bdc3c7"
    ours_bg = "#e8f6f3"     # Light teal
    ours_border = "#1abc9c"
    highlight_color = "#d35400" # Burnt orange for key concepts

    # ===========================
    # SECTION 1: PREVIOUS WORKS (Left Panel)
    # ===========================
    
    # Main panel rectangle
    rect_prev = patches.FancyBboxPatch((2, 15), 46, 80, boxstyle="round,pad=0.5", 
                                       ec=prev_work_border, fc=prev_work_bg, lw=2, zorder=1)
    ax.add_patch(rect_prev)

    # Header
    ax.text(25, 91, "Previous Benchmarks: Surface-Level or Elementary Insolvability", 
            ha='center', va='center', fontdict=header_font, color="#2c3e50")

    # Sub-headers
    ax.text(5, 87, "Example Question", fontdict=header_font, size=10, color="#7f8c8d")
    ax.text(35, 87, "Limitation / Nature of Trap", fontdict=header_font, size=10, color="#7f8c8d")

    # --- Row 1: Xue et al. ---
    y_pos = 78
    ax.text(5, y_pos, "(Xue et al. 2025)", weight='bold')
    q_xue = "Question: ...The three smallest triangular numbers that are also perfect squares are 1, 36, and 1225. What is the sum of the digits of the fourth smallest...?"
    ax.text(5, y_pos-2, q_xue, fontdict=example_font, wrap=True, width=28)
    
    anno_xue = "Limitation: Unverified Insolvability.\nRelying on missing standard definitions (e.g., 'triangular number') often retrievable by modern LLMs."
    ax.text(35, y_pos-1, anno_xue, wrap=True, width=18, va='top', size=9)
    
    # Separator line
    ax.plot([5, 45], [y_pos-9, y_pos-9], color=prev_work_border, lw=1, ls='--')

    # --- Row 2: Song et al. ---
    y_pos = 63
    ax.text(5, y_pos, "(Song et al. 2025)", weight='bold')
    q_song = "Question: The operation ⊗ is used to combine two nonzero numbers. Determine [(1 ⊗ 2) ⊗ 3] - [1 ⊗ (2 ⊗ 3)]."
    ax.text(5, y_pos-2, q_song, fontdict=example_font, wrap=True, width=28)

    anno_song = "Limitation: Common Sense / Trivial.\nInsolvability stems from a completely undefined symbol '⊗'. Lacks mathematical depth."
    ax.text(35, y_pos-1, anno_song, wrap=True, width=18, va='top', size=9)
    
    # Separator line
    ax.plot([5, 45], [y_pos-9, y_pos-9], color=prev_work_border, lw=1, ls='--')

    # --- Row 3: Sun et al. ---
    y_pos = 48
    ax.text(5, y_pos, "(Sun et al. 2024b)", weight='bold')
    q_sun = "Question: Tom had a total of 50 salty cookies and sweet cookies combined. He ate 14 sweet... and 9 salty... How many salty cookies did Tom have left?"
    ax.text(5, y_pos-2, q_sun, fontdict=example_font, wrap=True, width=28)

    anno_sun = "Limitation: Elementary Level.\nMissing basic initial condition (the split of 50 cookies). Requires no advanced reasoning."
    ax.text(35, y_pos-1, anno_sun, wrap=True, width=18, va='top', size=9)

    # ===========================
    # SECTION 2: OURS (Right Panel)
    # ===========================

    # Main panel rectangle
    rect_ours = patches.FancyBboxPatch((52, 15), 46, 80, boxstyle="round,pad=0.5", 
                                       ec=ours_border, fc=ours_bg, lw=2, zorder=1)
    ax.add_patch(rect_ours)

    # Header
    ax.text(75, 91, "MathTrap300 (Ours): Deep Mathematical Contradictions", 
            ha='center', va='center', fontdict=header_font, color="#16a085")
    
    # Sub-headers
    ax.text(55, 87, "Example Question (Functional Equation)", fontdict=header_font, size=10, color="#16a085")
    ax.text(85, 87, "Required Reasoning & Contradiction", fontdict=header_font, size=10, color="#16a085")

    # --- The Question ---
    y_pos = 80
    # Using mathtext for better rendering of equations
    q_ours_1 = r"Question: $f(2x)$ is symmetric about origin; $f(x+1)+f(3-x)=0$."
    q_ours_2 = r"For $x \in (2,4)$: $f(x)=-\log_{1/2}(x-2)+m$, with $m>0$."
    q_ours_3 = r"If $f(\frac{2022+\varepsilon-1}{2})=f(-1)$ for $\varepsilon \in (0,1)$, find smallest $m$."
    
    ax.text(55, y_pos, q_ours_1, fontdict=example_font)
    ax.text(55, y_pos-3, q_ours_2, fontdict=example_font)
    ax.text(55, y_pos-6, q_ours_3, fontdict=example_font)

    # --- The Reasoning Path (Visualizing the trap) ---
    # We will use arrows and text boxes to show the deductive steps leading to the contradiction.
    
    step_x = 82
    step_width = 22
    
    # Step 1: Symmetry & Periodicity
    ax.text(step_x, y_pos, "1. Derive Properties:", weight='bold', color=highlight_color, size=9)
    ax.text(step_x, y_pos-2, r"Symmetry $\to f(-x)=-f(x)$.", size=9)
    ax.text(step_x, y_pos-4, r"Functional Eq $\to f(x+4)=f(x)$ (Periodicity).", size=9)

    arrow_y1 = y_pos - 7
    ax.arrow(75, arrow_y1, 0, -2, head_width=1, head_length=1.5, fc=ours_border, ec=ours_border)

    # Step 2: Simplify Equation
    y_pos_s2 = arrow_y1 - 5
    ax.text(step_x, y_pos_s2, "2. Simplify Condition:", weight='bold', color=highlight_color, size=9)
    ax.text(step_x, y_pos_s2-2, r"Using period 4: $f(2022+\varepsilon) \to f(2+\varepsilon)$.", size=9)
    ax.text(step_x, y_pos_s2-4, r"Substitute given form: $f(2+\varepsilon) = \log_2\varepsilon+m$.", size=9)

    arrow_y2 = y_pos_s2 - 7
    ax.arrow(75, arrow_y2, 0, -2, head_width=1, head_length=1.5, fc=ours_border, ec=ours_border)

    # Step 3: The Contradiction (The Trap)
    y_pos_s3 = arrow_y2 - 5
    ax.text(step_x, y_pos_s3, "3. Identify Contradiction (The Trap):", weight='bold', color=highlight_color, size=9)
    ax.text(step_x, y_pos_s3-3, r"Final Eq becomes: $\log_2\varepsilon = m+1$.", size=9)
    
    # Highlight box for the contradiction
    contradict_box = patches.Rectangle((step_x-1, y_pos_s3-11), 23, 7, 
                                       fc='#fbeee6', ec=highlight_color, lw=2)
    ax.add_patch(contradict_box)
    ax.text(step_x, y_pos_s3-6.5, r"Constraint: $\varepsilon \in (0,1) \implies \log_2\varepsilon < 0$.", size=9, weight='bold')
    ax.text(step_x, y_pos_s3-9.5, r"Therefore $m < -1$." + "\n" + r"Please note that this Contradicts given condition $m>0$.", 
            size=9, weight='bold', color='#c0392b')


    # ===========================
    # SECTION 3: BOTTOM VISUAL ANCHOR (Knowledge Depth)
    # ===========================
    
    # Draw a large arrow across the bottom indicating increasing difficulty
    ax.arrow(10, 8, 80, 0, head_width=3, head_length=5, fc='#7f8c8d', ec='#7f8c8d', width=0.5)

    # Labels along the arrow
    ax.text(10, 3, "Elementary / Common Sense / Definitions", ha='center', size=11, weight='bold', color="#7f8c8d")
    ax.text(90, 3, "Deep Domain Knowledge / Multi-step Reasoning", ha='center', size=11, weight='bold', color="#16a085")
    ax.text(50, 10, "Required Mathematical Depth to Recognize Insolvability", ha='center', size=12, style='italic')


    # Final Title
    fig.suptitle("Figure 1: Comparison of Insolvable Problem Benchmarks.\nUnlike previous datasets relying on missing surface-level information, MathTrap300 requires detecting fundamental contradictions through deep mathematical reasoning.", 
                 fontsize=12, y=0.99, weight='bold')

    plt.tight_layout()
    # plt.savefig("figure1_mathtrap300.pdf", bbox_inches='tight') # Uncomment to save
    plt.show()

draw_figure_1()