package dev.ko2mc.blocks;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.data.BlockData;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.event.Event;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockBurnEvent;
import org.bukkit.event.block.BlockExplodeEvent;
import org.bukkit.event.block.BlockFadeEvent;
import org.bukkit.event.block.BlockFormEvent;
import org.bukkit.event.block.BlockFromToEvent;
import org.bukkit.event.block.BlockPhysicsEvent;
import org.bukkit.event.block.BlockPistonExtendEvent;
import org.bukkit.event.block.BlockPistonRetractEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.block.BlockRedstoneEvent;
import org.bukkit.event.block.BlockSpreadEvent;
import org.bukkit.event.block.LeavesDecayEvent;
import org.bukkit.event.block.NotePlayEvent;
import org.bukkit.event.entity.EntityChangeBlockEvent;
import org.bukkit.event.entity.EntityExplodeEvent;
import org.bukkit.event.entity.EntityInteractEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * ko2mc gives existing block states (note blocks, mushroom blocks, leaves, tripwire,
 * wool, ...) Knight Online textures through its resource pack. Vanilla Minecraft
 * changes some of those states by itself: a note block's instrument follows the block
 * under it, leaves track their distance to logs, mushroom blocks and tripwire react to
 * neighbours, redstone powers note blocks. Each state change would swap the texture.
 *
 * Whenever something changes in the world (a block placed, broken, exploded, pushed,
 * flowed over...), this plugin remembers the KO-textured blocks around it and, a few
 * ticks later, puts back any whose state changed while staying the same block type.
 * Breaking a KO-textured block still works normally.
 */
public final class KOBlocksPlugin extends JavaPlugin implements Listener {

    private static final int RADIUS = 2;
    private final Set<Material> hosts = EnumSet.noneOf(Material.class);
    private final Map<Location, BlockData> saved = new HashMap<>();
    private boolean restoreScheduled;
    private boolean enabled, noteBlocks, tripwire, leafDecay;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        for (String name : getConfig().getStringList("blocks")) {
            Material m = Material.matchMaterial(name);
            if (m != null && m.isBlock()) {
                hosts.add(m);
            } else {
                getLogger().warning("Unknown block in config: " + name);
            }
        }
        enabled = getConfig().getBoolean("enabled", true);
        noteBlocks = getConfig().getBoolean("protect-note-blocks", true);
        tripwire = getConfig().getBoolean("protect-tripwire", true);
        leafDecay = getConfig().getBoolean("no-leaf-decay", true);
        getServer().getPluginManager().registerEvents(this, this);
        getLogger().info("Protecting " + hosts.size() + " KO-textured block types");
    }

    private boolean isHost(Block b) {
        return hosts.contains(b.getType());
    }

    // ------------------------------------------------------------------ core

    /** Remember the KO blocks around a block that is about to change. */
    private void protectAround(Block center) {
        if (!enabled) return;
        World w = center.getWorld();
        int cx = center.getX(), cy = center.getY(), cz = center.getZ();
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dz = -RADIUS; dz <= RADIUS; dz++) {
                    if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > RADIUS + 1) continue;
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    int y = cy + dy;
                    if (y < w.getMinHeight() || y >= w.getMaxHeight()) continue;
                    Block b = w.getBlockAt(cx + dx, y, cz + dz);
                    if (isHost(b)) saved.putIfAbsent(b.getLocation(), b.getBlockData().clone());
                }
            }
        }
        scheduleRestore();
    }

    private void protectSelf(Block b) {
        if (enabled && isHost(b)) {
            saved.putIfAbsent(b.getLocation(), b.getBlockData().clone());
            scheduleRestore();
        }
    }

    private void scheduleRestore() {
        if (restoreScheduled) return;
        restoreScheduled = true;
        // leaves and scheduled block ticks update a tick or two later; check a few times
        getServer().getScheduler().runTaskLater(this, () -> restore(false), 1L);
        getServer().getScheduler().runTaskLater(this, () -> restore(false), 4L);
        getServer().getScheduler().runTaskLater(this, () -> restore(true), 12L);
    }

    private void restore(boolean last) {
        for (Map.Entry<Location, BlockData> e : saved.entrySet()) {
            Block b = e.getKey().getBlock();
            BlockData want = e.getValue();
            // same block type but a different state -> put the old state (texture) back
            if (b.getType() == want.getMaterial() && !b.getBlockData().equals(want)) {
                b.setBlockData(want, false);
            }
        }
        if (last) {
            saved.clear();
            restoreScheduled = false;
        }
    }

    // ------------------------------------------------------------------ world changes

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(BlockPlaceEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreak(BlockBreakEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBurn(BlockBurnEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onFade(BlockFadeEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onForm(BlockFormEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onSpread(BlockSpreadEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onFlow(BlockFromToEvent e) { protectAround(e.getToBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onEntityChange(EntityChangeBlockEvent e) { protectAround(e.getBlock()); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPistonExtend(BlockPistonExtendEvent e) {
        protectAround(e.getBlock());
        for (Block b : e.getBlocks()) protectAround(b.getRelative(e.getDirection()));
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPistonRetract(BlockPistonRetractEvent e) {
        protectAround(e.getBlock());
        for (Block b : e.getBlocks()) protectAround(b);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBlockExplode(BlockExplodeEvent e) { for (Block b : e.blockList()) protectAround(b); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onEntityExplode(EntityExplodeEvent e) { for (Block b : e.blockList()) protectAround(b); }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onRedstone(BlockRedstoneEvent e) { protectAround(e.getBlock()); }

    /** Any neighbour update reaching a KO block: keep its state. */
    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPhysics(BlockPhysicsEvent e) { protectSelf(e.getBlock()); }

    // ------------------------------------------------------------------ interaction

    /** Right-clicking a note block would tune it to the next note = the next texture. */
    @EventHandler(priority = EventPriority.HIGHEST)
    public void onInteract(PlayerInteractEvent e) {
        Block b = e.getClickedBlock();
        if (b == null) return;
        if (noteBlocks && e.getAction() == Action.RIGHT_CLICK_BLOCK && b.getType() == Material.NOTE_BLOCK) {
            e.setUseInteractedBlock(Event.Result.DENY);
        }
        if (tripwire && e.getAction() == Action.PHYSICAL && b.getType() == Material.TRIPWIRE) {
            e.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onNote(NotePlayEvent e) {
        if (noteBlocks) e.setCancelled(true);
    }

    /** Mobs walking through a tripwire (a KO plant sprite) would power it. */
    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onEntityInteract(EntityInteractEvent e) {
        if (tripwire && e.getBlock().getType() == Material.TRIPWIRE) e.setCancelled(true);
    }

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onDecay(LeavesDecayEvent e) {
        if (leafDecay && isHost(e.getBlock())) e.setCancelled(true);
    }

    // ------------------------------------------------------------------ admin/test command

    /** /ko2mc place x y z block  -- place a block the way a player would (for testing). */
    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 5 && args[0].equalsIgnoreCase("place")) {
            World w = getServer().getWorlds().get(0);
            Block b = w.getBlockAt(Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
            Material m = Material.matchMaterial(args[4]);
            if (m == null) {
                sender.sendMessage("Unknown block " + args[4]);
                return true;
            }
            protectAround(b);
            b.setType(m, true);
            sender.sendMessage("Placed " + m + " at " + b.getX() + " " + b.getY() + " " + b.getZ());
            return true;
        }
        sender.sendMessage("Usage: /ko2mc place <x> <y> <z> <block>");
        return true;
    }
}
