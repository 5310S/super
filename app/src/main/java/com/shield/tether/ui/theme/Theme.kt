package com.shield.tether.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val DarkColors = darkColorScheme(
    primary = BluePrimary,
    secondary = GreenAccent,
    background = BackgroundDark,
    surface = SurfaceDark,
    onPrimary = OnPrimary,
    onSecondary = OnPrimary,
    onBackground = OnBackground,
    onSurface = OnBackground
)

@Composable
fun ShieldTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColors,
        typography = Typography,
        content = content
    )
}
