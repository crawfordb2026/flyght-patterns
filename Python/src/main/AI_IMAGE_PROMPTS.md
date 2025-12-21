# AI Image Generation Prompts for Pipeline Diagram

## **Prompt 1: Clean Scientific Flowchart (Recommended)**

```
Create a professional scientific flowchart diagram showing a data processing pipeline. 
The diagram should have 5 horizontal boxes connected by arrows, arranged left to right:

Box 1 (left, blue): "Raw DAM Data" - contains small data point icons
Box 2 (green): "Data Processing" - shows clock/time icons, labeled "Align to ZT, Filter MT"
Box 3 (orange): "Sleep Detection" - shows sleep/moon icons, labeled "Identify 5-min bouts"
Box 4 (purple): "Feature Extraction" - shows chart/graph icons, labeled "21 Metrics"
Box 5 (right, dark blue): "Biological Insights" - shows lightbulb/insight icons

Style: Clean, modern, scientific presentation style. White background. 
Rounded rectangle boxes with subtle shadows. 
Arrows between boxes are simple and clean. 
Minimal text, professional color scheme. 
Suitable for academic presentation slide.
```

---

## **Prompt 2: Detailed Data Processing Pipeline (NEW - Recommended for First Slide)**

```
Create a comprehensive scientific data processing pipeline diagram for Drosophila sleep analysis.

Main vertical flow (top to bottom):

1. START: "Raw DAM Data" (blue box, left side)
   - Icon: database/spreadsheet
   - Subtext: "Monitor files, metadata"

2. "Data Processing & Filtering" (green box)
   - Main label: "Data Processing"
   - Show branching capabilities below or beside the box:
     • "Filter by Reading Type" (MT/CT/Pn) - filter/funnel icon
     • "Calculate ZT (Zeitgeber Time)" - clock icon, "Align to light cycle (ZT0-ZT24)"
     • "Filter by ZT Range" (optional) - time range icon, "Select specific hours"
     • "Exclude Days" (Days 1 & 7) - calendar with X icon, "Remove acclimation/washout"
     • "Calculate Exp_Day" - day counter icon
   
3. "Health Report Generation" (yellow/orange box)
   - Main label: "Health Classification"
   - Subtext: "Analyze fly health status"
   - Show capabilities:
     • "Calculate daily metrics" - chart icon
     • "Test startle response" - lightning icon
     • "Classify: Alive/Unhealthy/Dead/QC_Fail" - health check icon

4. "Fly Removal & Filtering" (red/orange box)
   - Main label: "Data Cleaning"
   - Subtext: "Remove problematic flies"
   - Show branching capabilities:
     • "Remove by Fly ID" (Monitor-Channel) - specific fly icon
     • "Remove by Genotype" - DNA icon
     • "Remove by Sex" - gender icons
     • "Remove by Treatment" - flask icon
     • "Remove Specific Days" (per fly) - calendar with X icon

5. "Sleep Detection" (purple box)
   - Main label: "Sleep Bout Detection"
   - Subtext: "Identify sleep periods"
   - Show: "5-minute inactivity threshold" - sleep/moon icon
   - Show: "Calculate ZT & Phase" (Light/Dark) - clock icon

6. END: "Clean Processed Data" (dark blue box, right side)
   - Icon: checkmark/success
   - Subtext: "Ready for feature extraction"

Visual Style:
- Main boxes are large rounded rectangles, connected by horizontal arrows
- Branching capabilities shown as smaller icons/text below or beside main boxes
- Use a clean, professional scientific diagram style
- White background
- Color progression: Blue → Green → Yellow → Red/Orange → Purple → Dark Blue
- Arrows clearly show data flow direction
- Text is clear and readable
- Suitable for academic presentation
- Layout should be balanced and not cluttered
- Note: Some steps can be done in different orders (e.g., health report can inform fly removal)
```

---

## **Prompt 3: Detailed Version with Labels**

```
Design a scientific data analysis pipeline flowchart for a biology research presentation.

Layout: Horizontal flow from left to right with 5 main stages:

1. "Raw DAM Data" (blue box) - Icon: database/spreadsheet
   - Subtext: "645,120 data points"
   
2. "Data Processing" (light green box) - Icon: gear/processing
   - Subtext: "Align to ZT, Filter MT"
   
3. "Sleep Detection" (orange box) - Icon: sleep/moon
   - Subtext: "5-minute inactivity bouts"
   
4. "Feature Extraction" (purple box) - Icon: bar chart
   - Subtext: "17 sleep + 4 circadian metrics"
   
5. "Biological Insights" (dark blue box) - Icon: lightbulb
   - Subtext: "Pattern discovery"

Visual style: Professional scientific diagram, clean and minimal. 
White background, modern flat design. 
Boxes have rounded corners, subtle gradients. 
Arrows are simple and directional. 
Text is clear and readable. 
Color palette: Blues, greens, oranges, purples - professional and accessible.
```

---

## **Prompt 4: Simple Minimalist Version**

```
Create a minimalist flowchart diagram for a scientific presentation:

Five connected boxes in a horizontal line:
1. Raw Data (blue)
2. Process (green) 
3. Detect Sleep (orange)
4. Extract Features (purple)
5. Insights (dark blue)

Style: Very clean, minimal design. Simple rounded rectangles. 
Thin arrows between boxes. 
White background. 
Professional scientific aesthetic. 
No decorative elements, just clear functional design.
```

---

## **Prompt 5: With Visual Metaphors**

```
Design a scientific pipeline diagram showing data transformation:

Left side: Raw data represented as scattered dots/points
↓ Arrow
Processing stage: Data being organized into rows/columns
↓ Arrow  
Sleep detection: Dots forming sleep patterns/waves
↓ Arrow
Feature extraction: Patterns becoming bar charts and graphs
↓ Arrow
Right side: Insights represented as lightbulbs or discovery icons

Style: Scientific illustration, clean and professional. 
Use visual metaphors to show transformation. 
Color progression from blue (data) to purple (insights). 
Suitable for biology research presentation.
```

---

## **Prompt 6: Vertical Flow Version**

```
Create a vertical scientific workflow diagram:

Top to bottom:
1. "Raw DAM Data" (blue box at top)
   ↓
2. "Data Processing" (green box)
   ↓
3. "Sleep Detection" (orange box)
   ↓
4. "Feature Extraction" (purple box)
   ↓
5. "Biological Insights" (dark blue box at bottom)

Each box should be a rounded rectangle with an icon.
Arrows flow downward.
Style: Clean scientific presentation diagram, white background.
Professional color scheme.
```

---

## **Usage Tips:**

1. **For DALL-E / ChatGPT Image Generation:**
   - Use Prompt 2 (most detailed for data processing)
   - May need to refine with follow-up prompts

2. **For Midjourney:**
   - Use Prompt 2, add: `--style scientific --v 6`
   - May need multiple iterations

3. **For Stable Diffusion:**
   - Use Prompt 2, add negative prompt: `"cluttered, messy, cartoon, artistic"`

4. **Best Results:**
   - Start with Prompt 2 (detailed data processing)
   - If result is too artistic, add: "technical diagram, flowchart style"
   - If result is too complex, simplify to Prompt 3
   - You can always edit in PowerPoint/Illustrator after generation

---

## **Alternative: Create in PowerPoint/Illustrator**

If AI generation doesn't work well, you can easily create this in PowerPoint:

1. Insert → SmartArt → Process → Basic Process (5 steps)
2. Customize colors: Blue → Green → Orange → Purple → Dark Blue
3. Add icons from Insert → Icons
4. Export as high-resolution PNG

This might be faster and more reliable than AI generation for a simple flowchart.
