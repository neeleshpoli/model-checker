#![windows_subsystem = "windows"]

#[cfg(windows)]
use windows_reactor::*;

#[cfg(windows)]
fn home_page(_cx: &mut RenderCx) -> Element {
    grid((
        // Window title bar
        TitleBar::new("ModelCheck")
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
            ScrollViewer::new(vstack((message("Test message to the model, it doesn't actually do anything but test the user interface"),)))
                .grid_row(1)
                .grid_column_span(0)
                .max_width(800.0),
            grid((
                // Textbox
                TextBox::new("")
                    .placeholder_text("Ask Gemma anything...")
                    .text_wrapping(TextWrapping::Wrap)
                    .horizontal_alignment(HorizontalAlignment::Stretch)
                    .multiline()
                    .grid_column(0),
                // Send button
                Button::new("Send")
                    .accent()
                    .vertical_alignment(VerticalAlignment::Bottom)
                    .grid_column(1),
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
    App::new()
        .title("ModelCheck")
        .backdrop(Backdrop::Mica)
        .inner_constraints(InnerConstraints {
            min_width: Some(600.0),
            min_height: Some(500.0),
            max_width: None,
            max_height: None,
        })
        .render(home_page)
}
