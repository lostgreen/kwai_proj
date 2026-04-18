<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import videojs from 'video.js'
import 'video.js/dist/video-js.css'

const props = defineProps<{
  src: string
}>()

const emit = defineEmits<{
  timeupdate: [time: number]
}>()

const videoRef = ref<HTMLVideoElement>()
let player: ReturnType<typeof videojs> | null = null

function seekTo(time: number) {
  player?.currentTime(time)
}

onMounted(() => {
  if (!videoRef.value) return
  player = videojs(videoRef.value, {
    controls: true,
    fluid: true,
    aspectRatio: '16:9',
    sources: [{ src: props.src, type: 'video/mp4' }],
  })
  player.on('timeupdate', () => {
    emit('timeupdate', player!.currentTime() ?? 0)
  })
})

watch(() => props.src, (newSrc) => {
  player?.src({ src: newSrc, type: 'video/mp4' })
})

onBeforeUnmount(() => {
  player?.dispose()
})

defineExpose({ seekTo })
</script>

<template>
  <div>
    <video ref="videoRef" class="video-js vjs-default-skin" />
  </div>
</template>
