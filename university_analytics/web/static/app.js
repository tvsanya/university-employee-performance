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
function validSource(target) {
  const file = target.querySelector('input[name="data"]');
  const address = target.querySelector('input[name="source_url"]');
  file.setCustomValidity(""); address.setCustomValidity("");
  const hasFile = file.files.length > 0, hasUrl = address.value.trim().length > 0;
  if (!hasFile && !hasUrl) file.setCustomValidity("Выберите CSV-файл или укажите ссылку.");
  if (hasFile && hasUrl) address.setCustomValidity("Оставьте один источник: файл или ссылку.");
  return target.reportValidity();
}
for (const target of document.querySelectorAll("#data-form, #training-form")) {
  target.addEventListener("input", () => {
    target.querySelector('input[name="data"]').setCustomValidity("");
    target.querySelector('input[name="source_url"]').setCustomValidity("");
  });
}
async function run(action) {
  if (!validSource(form)) return;
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
        note(`Помечено для просмотра: ${data.report.review_policy.review_rows}. Порог: ${data.report.review_policy.threshold}.`);
        const names = {source_row: "Строка CSV", predicted_class: "Прогноз", max_probability: "Максимальная вероятность", normalized_entropy: "Энтропия (0–1)", review_recommended: "Просмотр рекомендован"};
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
let activeJobId;
const cancelJob = document.querySelector("#cancel-job");
async function pollJob(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    if (sessionStorage.getItem("university-job-id")!==id) return;
    if (!response.ok) throw new Error(job.error || "Эксперимент недоступен.");
    activeJobId = id;
    cancelJob.hidden = !['running','queued','cancelling'].includes(job.state);
    cancelJob.disabled = job.state === 'cancelling';
    jobStage.textContent = job.stage;
    document.querySelector("#job-progress").value = job.progress;
    if (["running", "queued", "cancelling"].includes(job.state)) {
      jobMessage.textContent = "Можно продолжать работу со страницей. Статус обновляется автоматически.";
      jobTimer = setTimeout(() => pollJob(id), 1200);
    } else {
      trainingForm.querySelector("#train-submit").disabled = false;
      if (job.state === "failed") throw new Error(job.error);
      if (job.state === 'cancelled') {
        jobMessage.textContent = 'Эксперимент остановлен. Частичные результаты не выдаются за готовую модель.';
        document.querySelector('#job-metrics').replaceChildren();
        document.querySelector('#job-artifacts').replaceChildren();
        document.querySelector('#job-figures').replaceChildren();
        await refreshJobs(); return;
      }
      const label = {linear: "линейная", dummy: "константная", neural: "нейросеть"}[job.result.selected_model] || job.result.selected_model;
      jobMessage.textContent = `Выбрана модель: ${label}, по validation. Данные: ${job.result.data_origin}.`;
      if (job.result.research_status === "partial") jobMessage.textContent += " Некоторые анализы недоступны; причины сохранены в отчёте.";
      await showArtifacts(id);
      await refreshJobs();
      if (sessionStorage.getItem("university-job-id")!==id) return;
      const parts = ["train", "validation", "test"];
      table(document.querySelector("#job-metrics"), ["Выборка", "Accuracy", "Macro-F1"], parts.map(name => [name, job.result.metrics[name].accuracy.toFixed(4), job.result.metrics[name].macro_f1.toFixed(4)]));
    }
  } catch (error) {
    jobMessage.textContent = error.message;
    trainingForm.querySelector("#train-submit").disabled = false;
  }
}
trainingForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (!validSource(trainingForm)) return;
  clearTimeout(jobTimer);
  trainingForm.querySelector("#train-submit").disabled = true;
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
    trainingForm.querySelector("#train-submit").disabled = false;
  }
});
const previousJob = sessionStorage.getItem("university-job-id");
if (previousJob) { trainingForm.querySelector("#train-submit").disabled = true; pollJob(previousJob); }

async function showArtifacts(id) {
  const response = await fetch(`/api/jobs/${id}/artifacts`);
  const data = await response.json();
  if (sessionStorage.getItem("university-job-id")!==id) return;
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
let nextJobsCursor=null;
async function refreshJobs(before=null) {
  await predefinedReady;
  const response = await fetch("/api/jobs"+(before?"?before="+encodeURIComponent(before):""));
  const data = await response.json();
  if (!response.ok) return;
  const container = document.querySelector("#my-jobs"); container.replaceChildren();
  const select = document.querySelector("#experiment"), value = select.value;
  select.querySelectorAll('option[data-job]').forEach(option => { if(option.value!==value) option.remove(); });
  data.jobs.forEach(job => {
    const row = document.createElement("p"), open = document.createElement("button"); open.type = "button";
    row.dataset.jobId=job.id;
    open.textContent = new Date(job.created_at).toLocaleString("ru-RU") + " · " + job.stage;
    open.addEventListener("click", () => { clearTimeout(jobTimer); sessionStorage.setItem("university-job-id", job.id); pollJob(job.id); }); row.append(open); container.append(row);
    if (['completed','failed','cancelled'].includes(job.state)) {
      const remove=document.createElement('button');remove.type='button';remove.textContent='Удалить результат';
      remove.addEventListener('click',()=>deleteJob(job.id));row.append(remove);
    }
    if (job.state === "completed" && ![...select.options].some(option=>option.value==="job:"+job.id)) {
      const option = document.createElement("option"); option.value = "job:"+job.id; option.dataset.job = job.id;
      option.textContent = "Мой эксперимент · " + new Date(job.created_at).toLocaleString("ru-RU") + " · " + job.result.selected_model;
      select.append(option);
    }
  });
  if ([...select.options].some(option => option.value === value)) select.value = value;
  nextJobsCursor=data.next;
  document.querySelector("#next-jobs").hidden=!data.next;
  document.querySelector("#first-jobs").hidden=!before;
  if (!data.jobs.length) container.textContent = "После запуска здесь появятся ваши эксперименты.";
}
refreshJobs().catch(error => { document.querySelector("#my-jobs").textContent = error.message; });

const preflightButton = document.querySelector('#preflight');
const preflightResult = document.querySelector('#preflight-result');
trainingForm.addEventListener('input', () => { preflightResult.hidden = true; });
preflightButton.addEventListener('click', async () => {
  if (!validSource(trainingForm)) return;
  preflightButton.disabled = true; preflightResult.hidden = false;
  preflightResult.textContent = 'Проверка структуры и разбиения…';
  try {
    const response = await fetch('/api/train/validate', {method: 'POST', body: new FormData(trainingForm), headers: {'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось проверить настройки.');
    preflightResult.replaceChildren();
    const title = document.createElement('h3'); title.textContent = 'Настройки проверены'; preflightResult.append(title);
    const text = document.createElement('p');
    text.textContent = `${data.rows} строк, ${data.columns} колонок. Цель: ${data.target}. Признаков: ${data.features.length}. Данные: ${data.data_origin}. Эпох: ${data.epochs}.`;
    preflightResult.append(text);
    for (const [name, values] of Object.entries(data.partitions)) {
      const row = document.createElement('p'); row.textContent = `${name}: ${values.rows} строк, ${values.groups} независимых групп.`; preflightResult.append(row);
      if (data.class_counts?.[name]) {
        const counts = document.createElement('p'); counts.textContent = 'Классы: ' + Object.entries(data.class_counts[name]).map(([label, count]) => `${label}: ${count}`).join('; '); preflightResult.append(counts);
      }
    }
    for (const warning of [...data.warnings, data.limitation]) { const row = document.createElement('p'); row.textContent = warning; preflightResult.append(row); }
    const hash = document.createElement('p'); hash.textContent = `SHA-256: ${data.source_sha256}`; preflightResult.append(hash);
  } catch (error) { preflightResult.textContent = error.message; }
  finally { preflightButton.disabled = false; }
});

cancelJob.addEventListener('click', async () => {
  cancelJob.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/cancel`, {method:'POST', headers:{'X-CSRF-Token':document.querySelector('meta[name="csrf-token"]').content}});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось остановить обучение.');
    clearTimeout(jobTimer); await pollJob(activeJobId);
  } catch (error) { jobMessage.textContent=error.message; cancelJob.disabled=false; }
});

document.querySelector('#next-jobs').addEventListener('click',()=>refreshJobs(nextJobsCursor));
document.querySelector('#first-jobs').addEventListener('click',()=>refreshJobs());
async function deleteJob(id) {
  if (!window.confirm('Удалить исходный CSV и результаты этого эксперимента? Скачайте ZIP заранее, если он нужен.')) return;
  try {
    const response=await fetch(`/api/jobs/${id}/delete`,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':document.querySelector('meta[name="csrf-token"]').content},body:JSON.stringify({confirm:id})});
    const data=await response.json();if(!response.ok) throw new Error(data.error||'Не удалось удалить результат.');
    document.querySelector(`#experiment option[value="job:${id}"]`)?.remove();
    if(sessionStorage.getItem('university-job-id')===id){
      clearTimeout(jobTimer);sessionStorage.removeItem('university-job-id');activeJobId=null;cancelJob.hidden=true;
      jobStage.textContent='Эксперимент удалён';
      for(const name of ['#job-metrics','#job-artifacts','#job-figures']) document.querySelector(name).replaceChildren();
    }
    jobMessage.textContent=data.cleanup_pending?'Результат убран из списка; часть файлов ожидает очистки из-за ошибки файловой системы.':'Результат удалён.';
    await refreshJobs();
  } catch(error){jobMessage.textContent=error.message;}
}
