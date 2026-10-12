# Medical Imaging Report (Medical Disease Prediction)  
## Complete End-to-End Explanation of the Project

**Project title:** Medical Imaging Report — AI-Based Multi-Modality Disease Prediction  
**Developer:** Thirthaprasad C N  
**Repository:** https://github.com/Thirthaprasadcn/medical-disease-prediction  

This document explains, in continuous narrative form, how the entire project works from the moment a user opens the website until a finished diagnostic-style report is produced, reviewed, and optionally exported. Nothing essential in the working pipeline is skipped. All image sizes are stated in pixels. This system is a research and academic prototype. It is not a medical device, it is not clinically validated, and it must never be used to diagnose real patients.

---

### Introduction: what the project is trying to solve

Modern hospitals generate huge volumes of medical images every day, including brain MRI scans, chest X-rays, and chest CT studies. Reading these images carefully takes time, and different types of scans require different kinds of clinical knowledge. A single artificial-intelligence model trained only on chest X-rays cannot correctly interpret a brain MRI. If such a model is wrongly applied, it may produce a meaningless or dangerous label, for example calling a brain tumor image “pneumonia.”

This project was built to demonstrate a practical, industrial-style answer to that problem. It creates a complete local software system that accepts a medical image, prepares that image for analysis, decides which organ and modality it belongs to, runs the correct trained disease model, marks suspicious regions on the image, writes a structured report, and allows a human reviewer to accept or reject findings and sign the report. The same system can also export the report in FHIR format so that it could later connect to hospital information systems. The user experience is organized as three web pages: Home, Upload, and Output.

---

### The technology foundation

The backend of the project is written in Python and served through FastAPI and Uvicorn. Image loading and classical computer-vision steps use OpenCV, NumPy, and scikit-image. Deep learning inference uses PyTorch. The frontend is served as Jinja HTML templates with CSS and JavaScript, and a Three.js WebGL scene provides an industrial three-dimensional imaging atmosphere behind the pages. Reports are stored as JSON files on disk. Clinician review decisions are stored in a SQLite database. DICOM files can be ingested with pydicom. FHIR R4 bundles can be generated for interoperability. There is also a scaffolding layer for a simple PACS C-STORE receiver and a model registry with promote, rollback, and audit logging. Together these pieces form one continuous pipeline rather than a set of disconnected scripts.

---

### What happens when the user opens the website

When the server is started, the application listens on the local machine, typically at http://127.0.0.1:8000. On startup it initializes the review database and ensures a demo model registry entry exists. The first page the user sees is the Home page. This page introduces the product, shows basic status information such as how many reports have already been generated, and lists the disease models that are currently loaded. Those models include the brain tumor MRI classifier, the chest X-ray pneumonia classifier, the chest CT nodule classifier, and the brain MRI segmentation model. Each card shows whether the model is ready and reports its main metric where available.

From Home the user moves to the Upload page. This is the operational console of the system. Here the user chooses a file, selects the scan type, chooses clinical or demo mode, optionally sets a CT window, priority, reviewer name, and clinical note, and then starts analysis. A live pipeline indicator shows the stages of ingest, preprocess, inference, report, and review while the request is running. Behind the interface, a cinematic three-dimensional gantry and tomographic-slice animation reinforces the imaging theme without replacing the clinical content.

---

### Upload and study options in detail

The Upload page accepts PNG images, JPEG images, single DICOM instances with extensions such as .dcm, and ZIP archives containing DICOM files. The maximum allowed upload size is one hundred megabytes. The most important clinical control on this page is the scan type. The user may choose automatic detection, Brain MRI for tumor analysis, Chest X-ray for pneumonia analysis, or Chest CT for nodule analysis. Selecting the correct scan type is the safest way to guarantee that the right disease model is used. Mode may be set to clinical or demo. For DICOM CT studies, a window preset such as lung, soft tissue, or bone may be chosen so that the displayed slice uses an appropriate Hounsfield window before conversion to a two-dimensional image. Priority and reviewer fields are stored with the workflow metadata of the report for later audit and review.

When the user clicks analyze, the browser sends a multipart HTTP POST request to the API endpoint that creates a new scan. That request carries the file bytes together with the form fields for scan type, mode, window, priority, reviewer, and clinical note.

---

### Step one of the pipeline: ingestion

On the server, the first responsibility is to understand what kind of file arrived. If the file is a ZIP of DICOM instances or a single DICOM object, the DICOM ingestion module reads it with pydicom, extracts useful metadata such as modality and study identifiers, applies windowing when appropriate, converts one two-dimensional slice into PNG bytes, and passes that derived image into the same analysis path used for ordinary photographs. The original DICOM bytes may also be saved for later reference, and the study may be entered into a simple PACS-style queue.

If the file is a PNG or JPEG, the server validates the content type and size and then sends the raw image bytes directly into preprocessing. Empty files and unsupported types are rejected with clear error messages. At this point the system still does not know the disease label. It only knows that a valid imaging study has been accepted for analysis.

---

### Step two: preprocessing and the 512 by 512 working image

Every accepted study is forced through a common preprocessing stage so that later algorithms always see a predictable canvas. The uploaded bytes are decoded into a grayscale image. Color photographs are converted to a single intensity channel because the disease models in this project are trained on grayscale medical imagery.

Next, the image is resized while preserving aspect ratio so that its longer side becomes five hundred twelve pixels. The resized content is then placed onto a black square canvas of exactly five hundred twelve by five hundred twelve pixels. This letterboxing step means that a tall image and a wide image both become the same final shape without being stretched into distortion. That five hundred twelve by five hundred twelve grayscale matrix is called the original working image.

From the original working image the system creates an enhanced working image. Enhancement first applies a bilateral filter to reduce noise while trying to keep edges, and then applies CLAHE, which is contrast-limited adaptive histogram equalization, so that subtle structures become more visible for region detection. Both the original and the enhanced images remain five hundred twelve by five hundred twelve pixels with eight-bit unsigned integer intensities ranging from zero to two hundred fifty five.

This five hundred twelve pixel canvas is the master coordinate system of the project. Overlay circles, finding locations, and report quadrants are all expressed in that five hundred twelve by five hundred twelve space. Later, when a deep learning model needs a smaller input, the system temporarily resizes a copy of this canvas down to the model size, but findings are always mapped back to the five hundred twelve pixel grid for display.

---

### Step three: deciding anatomy and modality

After preprocessing, the system must choose which disease pathway to follow. This decision is made by combining the user’s scan type, DICOM modality if present, filename clues, and image appearance scores.

If the user explicitly selected Brain MRI, the study is treated as brain magnetic resonance imaging with high confidence. If the user selected Chest X-ray, the study is treated as a radiographic chest study. If the user selected Chest CT, the study is treated as computed tomography of the chest. These explicit choices override weaker automatic guesses and are the recommended way to run the system when the user already knows the scan type.

When automatic mode is used, DICOM modality tags are respected when available. An MR study is guided toward the brain path. A DX or CR study is guided toward the chest X-ray path. A CT study may be treated as chest CT, or as brain CT if the filename or appearance strongly suggests a head study. If there is no DICOM tag, the filename is inspected for words such as brain, tumor, glioma, meningitis-related terms, MRI, chest, pneumonia, lung, x-ray, CT, or nodule. Finally, if filename evidence is weak, heuristic scores estimate whether the image looks more like a brain MRI slice, which often has a dark background and a compact central organ silhouette, or more like a chest radiograph, which usually fills more of the frame with mid-tone tissue.

A critical design rule of this project is safety under ambiguity. Earlier versions could misclassify some brain tumor images as chest studies and then wrongly run the pneumonia model. The current router prefers the brain pathway when scores are close, and it refuses to apply the pneumonia classifier when anatomy cannot be trusted. Model-based arbitration can also consult the brain tumor network itself when heuristics disagree. The result of this stage is an anatomy package containing the chosen anatomy label, modality code, confidence, reason text, and numeric brain versus chest scores.

---

### Step four: modality routing into the correct disease path

The modality router is the central traffic controller of inference. Using the anatomy decision, it selects one of three main clinical paths and attaches the correct disease classifier and region analysis strategy.

On the brain path, the system uses the trained four-class brain tumor convolutional neural network for disease assessment. When the brain segmentation checkpoint is available, it also uses a compact U-Net-style network to propose tumor-region findings. If that clinical segmentation path is not usable for a particular case, the classical blob detector still proposes candidate regions on the enhanced five hundred twelve by five hundred twelve image. The report metadata records that this was a brain MRI path and that the disease model belongs to the brain-tumor family rather than to PneumoniaMNIST.

On the chest X-ray path, the system uses the PneumoniaMNIST convolutional neural network to decide between normal and pneumonia. Region findings are produced mainly by the classical detector on the five hundred twelve pixel canvas. This path is appropriate only for chest radiographs, not for MRI or CT brain images.

On the chest CT path, the system uses the trained CT nodule classifier to decide between nodule and no nodule. Findings may come from the clinical CT adapter, which can turn model confidence and image peaks into circular candidate regions, with classical detection available as a fallback. PneumoniaMNIST is intentionally not used on CT.

In every path the router returns the same overall package structure: a list of findings, an image-level disease classification dictionary, model information including anatomy reason and disease model name, the operating mode, the modality string, and the anatomy decision object. This common return shape allows the rest of the application to stay simple even though the internal models differ.

---

### The models in detail, including pixel sizes

The project does not rely on one universal network. It uses several specialized models, each with its own training data and input resolution.

The brain tumor classifier is a PyTorch convolutional network named BrainTumorCNN. After the study has been normalized to five hundred twelve by five hundred twelve pixels, a copy of that image is resized to sixty four by sixty four pixels for this model. The network receives a single grayscale channel with intensities scaled into the floating range from zero to one. It predicts one of four classes: no tumor, glioma tumor, meningioma tumor, or pituitary tumor. It was trained on a public brain MRI tumor classification dataset. Its reported test accuracy is approximately eighty three percent. On the report page the predicted class is shown in human-readable language, together with the probability and the full set of class probabilities.

The chest X-ray pneumonia classifier is a smaller binary convolutional network named SmallCNN. From the same five hundred twelve by five hundred twelve working image it receives a further resized copy of twenty eight by twenty eight pixels. That size matches the PneumoniaMNIST benchmark on which it was trained. The network outputs a single logit that is converted to a probability of pneumonia. Labels are normal and pneumonia. Its reported test accuracy is approximately eighty seven percent, with a high area under the ROC curve on the benchmark test split. This model answers only whether a chest radiograph looks more like the pneumonia class or the normal class of that public dataset. It is not valid for MRI or CT.

The chest CT nodule classifier is a convolutional network named CTNoduleCNN. It also consumes sixty four by sixty four grayscale inputs derived from the five hundred twelve pixel canvas. It was trained on center slices from NoduleMNIST3D, which is a public MedMNIST three-dimensional nodule benchmark used here as a practical proxy for LIDC-style CT nodule work. Its reported test accuracy is approximately eighty six percent. When the model suggests a nodule, the system also attempts to mark candidate locations on the five hundred twelve by five hundred twelve enhanced image so that the Output page can show an overlay.

The brain segmentation model is a compact U-Net-style network named MiniUNet. It takes a sixty four by sixty four grayscale patch and predicts a sixty four by sixty four probability mask of abnormal region. That mask is scaled back up to five hundred twelve by five hundred twelve so that connected components can be converted into finding circles in the same coordinate system as the rest of the report. It was trained on public brain MRI images using weak pseudo-masks derived from bright tissue inside the brain silhouette. Its reported Dice score is approximately seventy seven percent. This is a research proxy for BraTS-style segmentation rather than a full multi-modal BraTS nnU-Net deployment.

In addition to these deep models, every path can use a classical Laplacian-of-Gaussian blob detector that operates directly on the enhanced five hundred twelve by five hundred twelve image. It searches for both bright and dark focal candidates, scores them by contrast and size, removes duplicates, and keeps up to eight findings. Each finding carries an identifier, center coordinates, radius in pixels, polarity such as hyperintense or hypointense, and a confidence score. These findings become the circles drawn on the overlay image and the narrative bullets in the report.

There is also a model registry concept in the product. The registry can list, promote, and roll back approved model entries and keep an audit log. The website may show a registry default such as pneumonia_cnn, but the disease model roster shows all four ready pathways. During an actual upload, the modality router chooses the model for that study; the registry default alone does not force every image through pneumonia analysis.

---

### Step five: building the report and the overlay

Once findings and disease classification are ready, the report module creates a durable study record. A unique scan identifier is generated. An overlay image is drawn by converting the five hundred twelve by five hundred twelve grayscale original into a color image and drawing numbered circles around each finding, with colors that reflect confidence. Each finding also receives a short narrative sentence that mentions approximate diameter in pixels, the quadrant of the five hundred twelve by five hundred twelve field such as upper left or lower right, polarity relative to surrounding tissue, and a reminder that the mark is for review rather than a confirmed diagnosis. For brain studies the narrative language refers to brain parenchyma and possible focal lesion candidates.

The structured report JSON then stores the scan identifier, original filename, generation time, modality, model information, anatomy routing reason, workflow fields such as priority and reviewer and scan type, the full findings list, the image-level disease assessment, a summary impression, and a research disclaimer. The original image, the overlay image, and the JSON report are written into the local storage folders of the backend. Before the response is returned, the report is validated against an internal contract so that required fields are present.

The API then returns the report to the browser, and the Upload page redirects the user to the Output page for that scan identifier.

---

### Step six: the Output page and human review

The Output page is the clinical review surface of the project. It shows the original five hundred twelve by five hundred twelve image beside the AI overlay. The user can hide the overlay, invert the display, or change overlay opacity. The side panel presents the impression text, the detected modality, the anatomy path, the disease model that actually ran, and the routing reason. The disease assessment section shows the predicted disease label and probability. For the brain model it can also list probabilities for all four tumor-related classes.

Below that, each finding appears as a reviewable row. A clinician-style user may accept or reject each finding. Those decisions are stored in SQLite and remain associated with the scan. When review is complete, the user may sign off the report. The page also offers links to open the raw JSON and to download or view a FHIR R4 DiagnosticReport bundle. This human-in-the-loop stage is intentional. The software is designed to assist review, not to replace a licensed radiologist.

---

### Optional exports and hospital-facing pieces

Beyond the interactive report, the project includes several industrial extensions. FHIR export maps the internal report into a DiagnosticReport-oriented bundle so that the output can be discussed in interoperability terms. Feedback export can collect accepted and rejected findings for later retraining discussions. The model registry supports listing entries and promoting or rolling back an active registry model through a validation gate. PACS scaffolding can start a simple C-STORE SCP and maintain a short study queue for laboratory demonstration. Regulatory documentation under the docs folder outlines intended use thinking and high-level notes related to quality and risk frameworks. These pieces show how a student or research prototype can grow toward an industrial workflow, while still remaining honest that full hospital validation has not been performed.

---

### Walking through three complete examples

Consider a brain tumor PNG. The user selects Brain MRI as the scan type and uploads the file. The server decodes the image and converts it to five hundred twelve by five hundred twelve grayscale. Because the scan type is brain, the router selects the brain path. The brain tumor network receives a sixty four by sixty four copy and returns a class such as meningioma or glioma together with a probability. The segmentation network or the classical detector marks one or more regions on the five hundred twelve pixel canvas. The Output page therefore shows a brain disease assessment and must not show PneumoniaMNIST as the disease model.

Consider a chest X-ray. The user selects Chest X-ray and uploads the radiograph. After the same five hundred twelve by five hundred twelve preprocessing, the pneumonia network receives a twenty eight by twenty eight copy and returns normal or pneumonia. Findings are marked on the large canvas. The report clearly identifies the pneumoniamnist disease model.

Consider a chest CT study. The user selects Chest CT, or uploads a DICOM CT series that is windowed and converted to a slice. The five hundred twelve by five hundred twelve image is analyzed by the CT nodule network at sixty four by sixty four pixels. The disease assessment speaks about nodule versus no nodule, and the pneumonia model remains unused.

---

### Pixel sizes restated as one continuous idea

It is useful to remember the pixel story as a single chain. The user may upload an image of any reasonable width and height. The pipeline immediately standardizes that image to five hundred twelve by five hundred twelve pixels for detection, overlay drawing, and report geometry. From that master canvas, temporary smaller copies are created for neural networks: twenty eight by twenty eight pixels for pneumonia classification, and sixty four by sixty four pixels for brain tumor classification, CT nodule classification, and brain region segmentation. Segmentation masks predicted at sixty four by sixty four are scaled back to five hundred twelve by five hundred twelve before findings are extracted. In other words, five hundred twelve is the language of the report display, while twenty eight and sixty four are the languages of the trained networks.

---

### Limitations and honest scope

Even though the pipeline is complete, its scientific and regulatory limits must remain visible. The accuracies quoted above are measured on public research datasets and proxy tasks. They do not prove performance on a local hospital’s real case mix. The CT and BraTS-related components are lightweight trained proxies rather than full production nnDetection or nnU-Net systems trained on complete clinical volumes. The workflow is essentially two-dimensional slice analysis rather than full three-dimensional volumetric diagnosis. Every report carries a disclaimer that the output is for research and education, requires human review, and is not cleared for clinical use.

---

### Conclusion

From start to end, this project behaves as a guided imaging factory. A study enters through the website, is normalized to a five hundred twelve by five hundred twelve pixel working image, is classified by anatomy and modality, is analyzed by the matching disease model at either twenty eight or sixty four pixels, receives region marks on the large canvas, becomes a structured report with overlay, and finally waits for human acceptance, rejection, sign-off, and optional FHIR export. By separating brain, chest X-ray, and chest CT models, and by documenting their pixel sizes and roles clearly, the system demonstrates how an industrial medical imaging report console can be assembled as a coherent research prototype.

---

### How to run the system again

On Windows, the recommended approach is to use a Python virtual environment and start the FastAPI application from the project root with the repository on PYTHONPATH, then open the browser at http://127.0.0.1:8000. The helper script scripts/run.ps1 can automate the common local launch path. Uploading a study with the correct scan type is the most reliable way to see the intended disease model on the Output page.
