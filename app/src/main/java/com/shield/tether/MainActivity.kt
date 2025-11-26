package com.shield.tether

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.shield.tether.ui.theme.ShieldTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ShieldTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HotspotScreen()
                }
            }
        }
    }
}

@Composable
private fun HotspotScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()
    val snackbars = remember { SnackbarHostState() }

    var rootAvailable by remember { mutableStateOf<Boolean?>(null) }
    var busy by remember { mutableStateOf(false) }
    var logs by remember { mutableStateOf(listOf<String>()) }

    fun appendLog(message: String) {
        logs = (logs + message).takeLast(50)
    }

    fun runAsync(label: String, block: () -> CommandResult) {
        scope.launch(Dispatchers.IO) {
            busy = true
            appendLog("Running: $label")
            val result = block()
            appendLog("✔ $label -> ${result.output}")
            if (!result.success) {
                snackbars.showSnackbar("Failed: $label")
            }
            busy = false
        }
    }

    LaunchedEffect(Unit) {
        val detected = RootTools.isRootAvailable()
        rootAvailable = detected
        appendLog("Root ${if (detected) "detected" else "not detected"}")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = "Shield Hotspot",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Attempts to lift carrier hotspot limits by toggling provisioning flags and TTL rules. Root is required for automation; otherwise follow the manual steps below.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onBackground
        )

        HotspotActions(
            rootAvailable = rootAvailable,
            busy = busy,
            onRefreshRoot = {
                runAsync("check root") { CommandResult(RootTools.isRootAvailable(), "Checked root") }
            },
            onDisableProvisioning = {
            runAsync("disable provisioning") { RootTools.disableProvisioningChecks() }
            },
            onApplyTtlPatch = {
                runAsync("apply TTL patch") { RootTools.applyTtlPatch() }
            },
            onRevertTtlPatch = {
                runAsync("revert TTL patch") { RootTools.revertTtlPatch() }
            },
            onSetSystemTtl = {
                runAsync("set system TTL 65") { RootTools.setSystemDefaultTtl(65) }
            },
            onResetSystemTtl = {
                runAsync("reset system TTL") { RootTools.setSystemDefaultTtl(64) }
            },
            onDisableOffload = {
                runAsync("disable tether offload") { RootTools.disableOffload() }
            },
            onEnableOffload = {
                runAsync("enable tether offload") { RootTools.enableOffload() }
            }
        )

        NonRootHelpers(
            onOpenApn = { launchIntent(context, Settings.ACTION_APN_SETTINGS, "APN settings") },
            onOpenVpn = { launchIntent(context, Settings.ACTION_VPN_SETTINGS, "VPN settings") },
            onOpenTether = { launchIntent(context, Settings.ACTION_WIRELESS_SETTINGS, "Network settings") },
            onOpenWifi = { launchIntent(context, Settings.ACTION_WIFI_SETTINGS, "Wi‑Fi settings") }
        )

        ManualInstructions()

        Card(
            shape = CardDefaults.shape,
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.2f))
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = "Recent actions",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(modifier = Modifier.height(8.dp))
                if (logs.isEmpty()) {
                    Text(
                        text = "No actions yet.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    logs.asReversed().forEach { line ->
                        Text(
                            text = "• $line",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }

        SnackbarHost(hostState = snackbars)
    }
}

@Composable
private fun HotspotActions(
    rootAvailable: Boolean?,
    busy: Boolean,
    onRefreshRoot: () -> Unit,
    onDisableProvisioning: () -> Unit,
    onApplyTtlPatch: () -> Unit,
    onRevertTtlPatch: () -> Unit,
    onSetSystemTtl: () -> Unit,
    onResetSystemTtl: () -> Unit,
    onDisableOffload: () -> Unit,
    onEnableOffload: () -> Unit
) {
    Card(
        shape = CardDefaults.shape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Root status: ${rootAvailable?.let { if (it) "available" else "missing" } ?: "checking..."}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Button(
                    onClick = onRefreshRoot,
                    enabled = !busy
                ) {
                    Text("Recheck")
                }
            }
            Button(
                onClick = onDisableProvisioning,
                enabled = !busy && (rootAvailable != false)
            ) {
                Text("Disable provisioning check")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onApplyTtlPatch,
                    enabled = !busy && (rootAvailable != false)
                ) {
                    Text("Apply TTL patch")
                }
                Button(
                    onClick = onRevertTtlPatch,
                    enabled = !busy && (rootAvailable != false)
                ) {
                    Text("Revert TTL patch")
                }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = onSetSystemTtl,
                    enabled = !busy && (rootAvailable != false),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Set system TTL=65")
                }
                Button(
                    onClick = onResetSystemTtl,
                    enabled = !busy && (rootAvailable != false),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Reset system TTL")
                }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = onDisableOffload,
                    enabled = !busy && (rootAvailable != false),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Disable offload")
                }
                Button(
                    onClick = onEnableOffload,
                    enabled = !busy && (rootAvailable != false),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Enable offload")
                }
            }
        }
    }
}

@Composable
private fun ManualInstructions() {
    Card(
        shape = CardDefaults.shape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.2f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Manual steps (if root is unavailable)",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "1) Connect via USB and run: adb shell settings put global tether_dun_required 0",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "2) On some carriers: adb shell settings put global tether_entitlement_check_state 0",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "3) To disguise tethering, set your router TTL to 65 or run iptables TTL rules on your phone (root needed).",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "4) Disable tether offload (adb shell settings put global tether_offload_disabled 1) if TTL rules aren’t sticking.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "5) Always-on VPN on the phone can hide traffic patterns/hostnames from the carrier; share the VPN-using data via hotspot.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "6) Use the same APN as normal phone data (no dedicated DUN APN) and avoid obvious P2P/streaming spikes when testing.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "7) On your router/client, prefer HTTPS and avoid P2P to reduce traffic fingerprints; cap speeds if possible.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "Changes may reset after reboot; re-run if hotspot reuse stops working. Carrier terms may still apply.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun NonRootHelpers(
    onOpenApn: () -> Unit,
    onOpenVpn: () -> Unit,
    onOpenTether: () -> Unit,
    onOpenWifi: () -> Unit
) {
    Card(
        shape = CardDefaults.shape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.secondary.copy(alpha = 0.2f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Non-root shortcuts",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "Use these to align with normal phone behavior: pick normal internet APN, enable phone-wide VPN, and review tether/Wi‑Fi toggles.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(onClick = onOpenApn, modifier = Modifier.weight(1f)) { Text("APN settings") }
                Button(onClick = onOpenVpn, modifier = Modifier.weight(1f)) { Text("VPN settings") }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(onClick = onOpenTether, modifier = Modifier.weight(1f)) { Text("Network settings") }
                Button(onClick = onOpenWifi, modifier = Modifier.weight(1f)) { Text("Wi‑Fi settings") }
            }
        }
    }
}

private fun launchIntent(context: android.content.Context, action: String, label: String) {
    try {
        val intent = Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
    } catch (e: Exception) {
        Toast.makeText(context, "Cannot open $label: ${e.message}", Toast.LENGTH_SHORT).show()
    }
}

data class CommandResult(val success: Boolean, val output: String)

private object RootTools {
    fun isRootAvailable(): Boolean {
        val check = runAsRoot("id")
        return check.success
    }

    fun disableProvisioningChecks(): CommandResult {
        val commands = listOf(
            "setprop net.tethering.noprovisioning true",
            "settings put global tether_dun_required 0",
            "settings put global tether_entitlement_check_state 0"
        )
        return runAsRoot(commands.joinToString(" && "))
    }

    fun applyTtlPatch(): CommandResult {
        val cmd = """
            iptables -t mangle -C POSTROUTING -j TTL --ttl-set 65 2>/dev/null || iptables -t mangle -A POSTROUTING -j TTL --ttl-set 65
        """.trimIndent()
        return runAsRoot(cmd)
    }

    fun revertTtlPatch(): CommandResult {
        val cmd = """
            iptables -t mangle -D POSTROUTING -j TTL --ttl-set 65 2>/dev/null
        """.trimIndent()
        return runAsRoot(cmd)
    }

    fun setSystemDefaultTtl(ttl: Int): CommandResult {
        // Try both procfs and sysctl; one of them usually works on rooted builds.
        val cmd = """
            (echo $ttl > /proc/sys/net/ipv4/ip_default_ttl 2>/dev/null) || sysctl -w net.ipv4.ip_default_ttl=$ttl
        """.trimIndent()
        return runAsRoot(cmd)
    }

    fun disableOffload(): CommandResult {
        val commands = listOf(
            "settings put global tether_offload_disabled 1",
            "setprop tether.offload.enabled false",
            "setprop persist.vendor.data.iwlan.enable false"
        )
        return runAsRoot(commands.joinToString(" && "))
    }

    fun enableOffload(): CommandResult {
        val commands = listOf(
            "settings delete global tether_offload_disabled",
            "setprop tether.offload.enabled true",
            "setprop persist.vendor.data.iwlan.enable true"
        )
        return runAsRoot(commands.joinToString(" && "))
    }

    private fun runAsRoot(command: String): CommandResult {
        return try {
            val process = ProcessBuilder("su", "-c", command)
                .redirectErrorStream(true)
                .start()

            val output = process.inputStream.bufferedReader().use { it.readText() }.trim()
            val success = process.waitFor() == 0
            CommandResult(success, output.ifEmpty { "no output" })
        } catch (e: Exception) {
            CommandResult(false, e.message ?: "Unknown error")
        }
    }
}
