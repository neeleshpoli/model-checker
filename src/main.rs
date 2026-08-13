#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
mod model_manager;

#[cfg(windows)]
use tokio::sync::{
    mpsc::{self, UnboundedSender},
    watch,
};
#[cfg(windows)]
use windows_reactor::*;

#[cfg(windows)]
use crate::model_manager::BackendCommands;
use crate::model_manager::ModelManager;

#[cfg(windows)]
fn home_page(
    cx: &mut RenderCx,
    backend_tx: UnboundedSender<BackendCommands>,
    ep_progress: watch::Receiver<f64>,
    model_progress: watch::Receiver<f64>,
) -> Element {
    let (textbox, set_textbox) = cx.use_state(String::new());
    let (query, set_query) = cx.use_state(String::new());
    let (response, set_response) = cx.use_async_state(String::new());

    let (ready, set_ready) = cx.use_async_state(false);

    let backend_temp = backend_tx.clone();

    grid((
        // Window title bar
        TitleBar::new("ModelChecker")
            .subtitle("Qwen 2.5")
            .grid_row(0)
            .grid_column(0),
        grid((
            // Model name
            hstack((
                TextBlock::new("Qwen 2.5").font_size(28.0).semibold(),
                model_status(cx, ep_progress, model_progress, set_ready),
            ))
            .spacing(8.0)
            .grid_row(0)
            .grid_column(0),
            // Messages and responses
            ScrollViewer::new(
                vstack((message(&query), TextBlock::new(&response).selectable())).spacing(12.0),
            )
            .grid_row(1)
            .grid_column_span(0)
            .max_width(800.0)
            .padding(8.0),
            grid((
                // Textbox
                TextBox::new(&textbox)
                    .placeholder_text("Ask Qwen anything...")
                    .text_wrapping(TextWrapping::Wrap)
                    .horizontal_alignment(HorizontalAlignment::Stretch)
                    .multiline()
                    .on_text_changed(move |c: String| {
                        set_textbox.call(c.clone());
                    })
                    .max_height(100.0)
                    .grid_column(0),
                // Send button
                Button::new("Send")
                    .accent()
                    .enabled(ready)
                    .vertical_alignment(VerticalAlignment::Bottom)
                    .grid_column(1)
                    .on_click(move || {
                        // Capture the current text directly
                        let current_text = textbox.clone();
                        set_query.call(current_text.clone());

                        let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();

                        backend_temp
                            .send(BackendCommands::Query {
                                query: current_text,
                                sender: response_tx,
                            })
                            .unwrap();

                        let set_response_clone = set_response.clone();

                        // Spawn a background task to await chunks asynchronously without blocking the UI
                        tokio::spawn(async move {
                            let mut full_response = String::new();

                            while let Some(recv_response) = response_rx.recv().await {
                                // Append to a local accumulator, then update the state
                                full_response.push_str(&recv_response);
                                set_response_clone.call(full_response.clone());
                            }
                        });
                    }),
            ))
            .columns([GridLength::Star(1.0), GridLength::Auto])
            .column_spacing(8.0)
            .min_width(500.0)
            .max_width(750.0)
            .grid_row(2)
            .grid_column(0),
        ))
        .rows([GridLength::Auto, GridLength::Star(1.0), GridLength::Auto])
        .columns([GridLength::Star(1.0)])
        .padding(Thickness {
            left: 12.0,
            top: 0.0,
            right: 12.0,
            bottom: 12.0,
        })
        .grid_row(1)
        .grid_column(0),
    ))
    .rows([GridLength::Auto, GridLength::Star(1.0)])
    .columns([GridLength::Star(1.0)])
    .into()
}

fn model_status(
    cx: &mut RenderCx,
    ep_progress: watch::Receiver<f64>,
    model_progress: watch::Receiver<f64>,
    set_ready: AsyncSetState<bool>,
) -> Element {
    let (ep_progress_state, set_ep_progress) = cx.use_async_state(0.0f64);
    let (model_progress_state, set_model_progress) = cx.use_async_state(0.0f64);

    let mut ep_progress_value = ep_progress.clone();
    let mut model_progress_value = model_progress.clone();
    tokio::spawn(async move {
        while let Ok(_) = ep_progress_value.changed().await {
            set_ep_progress.call(*ep_progress_value.borrow());
        }
        while let Ok(_) = model_progress_value.changed().await {
            set_model_progress.call(*model_progress_value.borrow());
        }

        set_ready.call(true);
    });

    // Check if the watch sender has closed, indicating what stage the backed is on
    if model_progress.has_changed().is_err() {
        // Both channels are closed; everything is done downloading
        TextBlock::new("Model ready to be used")
            .vertical_alignment(VerticalAlignment::Center)
            .into()
    } else if ep_progress.has_changed().is_err() {
        // EP sender dropped, but model is still downloading
        hstack((
            TextBlock::new("Downloading model"),
            ProgressBar::new(model_progress_state).width(200.0),
        ))
        .spacing(8.0)
        .vertical_alignment(VerticalAlignment::Center)
        .into()
    } else {
        hstack((
            TextBlock::new("Downloading execution providers"),
            ProgressBar::new(ep_progress_state).width(200.0),
        ))
        .spacing(8.0)
        .vertical_alignment(VerticalAlignment::Center)
        .into()
    }
}

#[cfg(windows)]
fn message(message: &str) -> Element {
    Border::new(TextBlock::new(message).wrap().selectable().max_width(500.0))
        .background(ThemeRef::CardBackground)
        .padding(12.0)
        .corner_radius(8.0)
        .horizontal_alignment(HorizontalAlignment::Right)
        .into()
}

#[cfg(windows)]
fn main() -> Result<()> {
    bootstrap()?;

    // In order to make sure the UI doesn't hang, we will
    // keep it blocking and ensure it is outside of the
    // async runtime
    let rt = tokio::runtime::Builder::new_multi_thread().build()?;
    let _gaurd = rt.enter();

    let (backend_tx, mut backend_rx) = mpsc::unbounded_channel();

    let (ep_progress_tx, ep_progress_rx) = watch::channel(0f64);
    let (model_progress_tx, model_progress_rx) = watch::channel(0f64);

    rt.spawn(async move {
        let model = ModelManager::new(ep_progress_tx, model_progress_tx)
            .await
            .unwrap();

        while let Some(command) = backend_rx.recv().await {
            match command {
                BackendCommands::Query { query, sender } => {
                    model.ask(query, sender).await.unwrap();
                }
                BackendCommands::_InitializeModel {
                    ep_download: _,
                    model_download: _,
                } => {}
            }
        }
    });

    App::new()
        .title("ModelChecker")
        .backdrop(Backdrop::Mica)
        .inner_constraints(InnerConstraints {
            min_width: Some(600.0),
            min_height: Some(500.0),
            max_width: None,
            max_height: None,
        })
        .render(move |cx| {
            home_page(
                cx,
                backend_tx.clone(),
                ep_progress_rx.clone(),
                model_progress_rx.clone(),
            )
        })?;

    Ok(())
}
