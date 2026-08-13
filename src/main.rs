#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]
mod model_manager;

#[cfg(windows)]
use tokio::sync::mpsc::{self, UnboundedSender};
#[cfg(windows)]
use windows_reactor::*;

#[cfg(windows)]
use crate::model_manager::BackendCommands;
use crate::model_manager::ModelManager;

#[cfg(windows)]
fn home_page(cx: &mut RenderCx, backend_tx: UnboundedSender<BackendCommands>) -> Element {
    let (textbox, set_textbox) = cx.use_state(String::new());
    let (query, set_query) = cx.use_state(String::new());
    let (response, set_response) = cx.use_async_state(String::new());

    let backend_temp = backend_tx.clone();

    grid((
        // Window title bar
        TitleBar::new("ModelChecker")
            .subtitle("Gemma 4")
            .grid_row(0)
            .grid_column(0),
        grid((
            // Model name
            TextBlock::new("Gemma 4")
                .font_size(28.0)
                .semibold()
                .grid_row(0)
                .grid_column(0),
            // Messages and responses
            ScrollViewer::new(vstack((message(&query), TextBlock::new(&response))).spacing(12.0))
                .grid_row(1)
                .grid_column_span(0)
                .max_width(800.0)
                .padding(8.0),
            grid((
                // Textbox
                TextBox::new(&textbox)
                    .placeholder_text("Ask Gemma anything...")
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

                        // 3. Spawn a background task to await chunks asynchronously without blocking the UI
                        tokio::spawn(async move {
                            let mut full_response = String::new();

                            while let Some(recv_response) = response_rx.recv().await {
                                // 4. Append to a local accumulator, then update the state
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

    let (backend_tx, mut backend_rx) = mpsc::unbounded_channel::<BackendCommands>();

    rt.spawn(async move {
        let model = ModelManager::new(
            |name, progress| println!("{name} download {progress}%"),
            |progress| println!("Model download progress {progress}%"),
        )
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
        .render(move |cx| home_page(cx, backend_tx.clone()))?;

    Ok(())
}
