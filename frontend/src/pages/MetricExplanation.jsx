// Business-friendly glossary. Definitions come from the backend glossary, with
// a curated order and short "why it matters" notes layered on top.
const ORDER = [
  ["pr_auc", "PR-AUC", "Primary metric — best for rare classes."],
  ["recall_declined", "Recall (Declined)", "Are we catching the declines?"],
  ["precision_declined", "Precision (Declined)", "Are our decline alerts trustworthy?"],
  ["f1_declined", "F1 (Declined)", "Balance of the two above."],
  ["balanced_accuracy", "Balanced Accuracy", "Fair to both classes."],
  ["roc_auc", "ROC-AUC", "Overall ranking quality."],
  ["confusion_matrix", "Confusion Matrix", "Where the errors land."],
  ["accuracy", "Accuracy", "De-emphasised — misleading here."],
];

export default function MetricExplanation({ data }) {
  const g = data.metric_glossary || {};
  return (
    <div>
      <h1>What the Metrics Mean</h1>
      <p className="subtitle">
        Plain-English definitions so non-technical readers can judge the model fairly.
      </p>

      <div className="warn-box">
        Why not just Accuracy? Only ~{((data.overview.class_counts.Declined / data.overview.n_rows) * 100).toFixed(0)}%
        of claims are Declined. A model that always says “Completed” would score ~
        {(100 - (data.overview.class_counts.Declined / data.overview.n_rows) * 100).toFixed(0)}% accuracy
        while catching <strong>zero</strong> declines. That is why we lead with PR-AUC, Recall, Precision and F1.
      </div>

      <div className="panel">
        {ORDER.filter(([k]) => g[k]).map(([k, term, why]) => (
          <div key={k} className="glossary-item">
            <div className="term">{term}</div>
            <div className="explain">{g[k].replace(/^[^:]+:\s*/, "")}</div>
            <div className="note">{why}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
