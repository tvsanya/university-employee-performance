"use strict";
const form = document.querySelector("#data-form");
const statusNode = document.querySelector("#status");
const summary = document.querySelector("#summary");
const heading = document.querySelector("#result-heading");
const download = document.querySelector("#download");
let outputCsv = "";
function table(container, headers, rows) {
  const result = document.createElement("table");
  const head = result.createTHead().insertRow();
  headers.forEach(label => { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = label; head.append(cell); });
  const body = result.createTBody();
  rows.forEach(row => { const tr = body.insertRow(); row.forEach(value => { tr.insertCell().textContent = value ?? "—"; }); });
  container.replaceChildren(result);
}
function note(text) { const node = document.createElement("p"); node.textContent = text; summary.append(node); }
async function run(action) {
  if (!form.reportValidity()) return;
  const buttons = form.querySelectorAll("button");
  buttons.forEach(button => { button.disabled = true; });
  statusNode.textContent = "Обработка файла…";
  statusNode.classList.remove("error"); summary.replaceChildren(); document.querySelector("#table").replaceChildren();
  outputCsv = ""; download.hidden = true;
  try {
    const response = await fetch(`/api/${action}`, {method: "POST", body: new FormData(form), headers: {"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не удалось обработать файл.");
    if (action === "inspect") {
      heading.textContent = "Профиль данных";
      statusNode.textContent = `Строк: ${data.row_count}. Колонок: ${data.column_count}. Полных повторов: ${data.duplicate_rows}.`;
      note(`Происхождение: ${data.data_origin === "synthetic" ? "симуляционные данные" : "по файлу не подтверждено"}.`);
      table(document.querySelector("#table"), ["Колонка", "Тип", "Пропуски", "NaN / ∞", "Разных значений"], Object.entries(data.column_profiles).map(([name, item]) => [name, item.kind === "numeric" ? "Число" : "Категория", item.missing, item.nonfinite, item.unique]));
    } else {
      heading.textContent = "Прогнозы готовы";
      statusNode.textContent = `Обработано строк: ${data.report.rows}. Ниже первые ${data.preview.length}; полный результат доступен в CSV.`;
      note(data.report.intended_use === "development_demo_only" ? "Демонстрационная модель на симуляции. Результаты не являются реальной оценкой сотрудников." : "Пригодность модели для реальных решений требует отдельной проверки.");
      if (data.preview.length) {
        const keys = Object.keys(data.preview[0]);
        const names = {source_row: "Строка CSV", predicted_class: "Прогноз"};
        const labels = keys.map(key => names[key] || (data.report.probability_columns[key] ? "Вероятность: " + data.report.probability_columns[key] : key));
        table(document.querySelector("#table"), labels, data.preview.map(row => keys.map(key => row[key])));
      }
      outputCsv = data.csv; download.hidden = false;
    }
  } catch (error) { statusNode.textContent = error.message; statusNode.classList.add("error"); }
  finally { buttons.forEach(button => { button.disabled = false; }); }
}
form.addEventListener("submit", event => { event.preventDefault(); run("predict"); });
document.querySelector("#inspect").addEventListener("click", () => run("inspect"));
download.addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob(["\ufeff", outputCsv], {type: "text/csv;charset=utf-8"}));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = "predictions.csv"; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const predefinedReady = (async () => {
  try {
    const response = await fetch("/api/experiments");
    if (!response.ok) throw new Error("Не удалось загрузить эксперименты.");
    const data = await response.json();
    const select = document.querySelector("#experiment"); select.replaceChildren();
    const rows = [], names = {dummy: "Константная", linear: "Линейная", neural: "Нейросеть"};
    data.experiments.forEach(item => {
      if (item.has_portable) { const option = document.createElement("option"); option.value = item.id; option.textContent = item.id + (item.data_origin === "synthetic" ? " · симуляция" : ""); select.append(option); }
      Object.entries(item.metrics).forEach(([name, score]) => rows.push([names[name] || name, item.data_origin, score.validation.macro_f1?.toFixed(4), score.test.macro_f1?.toFixed(4)]));
    });
    if (!select.options.length) { const option = document.createElement("option"); option.value = ""; option.textContent = "Загрузите собственную JSON-модель"; select.append(option); }
    table(document.querySelector("#experiments"), ["Модель", "Данные", "Validation macro-F1", "Test macro-F1"], rows);
  } catch (error) { document.querySelector("#experiments").textContent = error.message; }
})();

const trainingForm = document.querySelector("#training-form");
const jobStage = document.querySelector("#job-stage");
const jobMessage = document.querySelector("#job-message");
let jobTimer;
async function pollJob(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "Эксперимент недоступен.");
    jobStage.textContent = job.stage;
    document.querySelector("#job-progress").value = job.progress;
    if (["running", "queued"].includes(job.state)) {
      jobMessage.textContent = "Можно продолжать работу со страницей. Статус обновляется автоматически.";
      jobTimer = setTimeout(() => pollJob(id), 1200);
    } else {
      trainingForm.querySelector("button").disabled = false;
      if (job.state === "failed") throw new Error(job.error);
      const label = {linear: "линейная", dummy: "константная", neural: "нейросеть"}[job.result.selected_model] || job.result.selected_model;
      jobMessage.textContent = `Выбрана модель: ${label}, по validation. Данные: ${job.result.data_origin}.`;
      await showArtifacts(id);
      await refreshJobs();
      const parts = ["train", "validation", "test"];
      table(document.querySelector("#job-metrics"), ["Выборка", "Accuracy", "Macro-F1"], parts.map(name => [name, job.result.metrics[name].accuracy.toFixed(4), job.result.metrics[name].macro_f1.toFixed(4)]));
    }
  } catch (error) {
    jobMessage.textContent = error.message;
    trainingForm.querySelector("button").disabled = false;
  }
}
trainingForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (!trainingForm.reportValidity()) return;
  clearTimeout(jobTimer);
  trainingForm.querySelector("button").disabled = true;
  jobMessage.textContent = "Проверка входных данных…";
  document.querySelector("#job-metrics").replaceChildren();
  document.querySelector("#job-artifacts").replaceChildren();
  try {
    const response = await fetch("/api/train", {method: "POST", body: new FormData(trainingForm), headers: {"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content}});
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "Не удалось начать обучение.");
    sessionStorage.setItem("university-job-id", job.id);
    await pollJob(job.id);
  } catch (error) {
    jobMessage.textContent = error.message;
    trainingForm.querySelector("button").disabled = false;
  }
});
const previousJob = sessionStorage.getItem("university-job-id");
if (previousJob) { trainingForm.querySelector("button").disabled = true; pollJob(previousJob); }

async function showArtifacts(id) {
  const response = await fetch(`/api/jobs/${id}/artifacts`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Результаты недоступны.");
  const links = document.querySelector("#job-artifacts"); links.replaceChildren();
  data.files.forEach(item => {
    const link = document.createElement("a");
    link.href = `/api/jobs/${id}/download/${encodeURIComponent(item.name)}`;
    link.textContent = item.label; links.append(link);
  });
  const apply = document.createElement("button"); apply.type = "button"; apply.textContent = "Выбрать для прогноза";
  apply.addEventListener("click", async () => { await refreshJobs(); document.querySelector("#experiment").value = "job:"+id; form.scrollIntoView({behavior: "smooth"}); }); links.append(apply);
  const figures = document.querySelector("#job-figures"); figures.replaceChildren();
  ["training.png", "neural-confusion.png"].forEach(name => {
    const figure = document.createElement("figure"); const image = document.createElement("img");
    image.src = `/api/jobs/${id}/figure/${name}`; image.alt = name === "training.png" ? "История обучения нейросети текущего эксперимента" : "Матрица ошибок нейросети текущего эксперимента";
    const caption = document.createElement("figcaption"); caption.textContent = image.alt;
    figure.append(image, caption); figures.append(figure);
  });
}
async function refreshJobs() {
  await predefinedReady;
  const response = await fetch("/api/jobs");
  const data = await response.json();
  if (!response.ok) return;
  const container = document.querySelector("#my-jobs"); container.replaceChildren();
  const select = document.querySelector("#experiment"), value = select.value;
  select.querySelectorAll('option[data-job]').forEach(option => option.remove());
  data.jobs.forEach(job => {
    const row = document.createElement("p"), open = document.createElement("button"); open.type = "button";
    open.textContent = new Date(job.created_at).toLocaleString("ru-RU") + " · " + job.stage;
    open.addEventListener("click", () => { clearTimeout(jobTimer); sessionStorage.setItem("university-job-id", job.id); pollJob(job.id); }); row.append(open); container.append(row);
    if (job.state === "completed") {
      const option = document.createElement("option"); option.value = "job:"+job.id; option.dataset.job = job.id;
      option.textContent = "Мой эксперимент · " + new Date(job.created_at).toLocaleString("ru-RU") + " · " + job.result.selected_model;
      select.append(option);
    }
  });
  if ([...select.options].some(option => option.value === value)) select.value = value;
  if (!data.jobs.length) container.textContent = "После запуска здесь появятся ваши эксперименты.";
}
refreshJobs().catch(error => { document.querySelector("#my-jobs").textContent = error.message; });
