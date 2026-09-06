import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发期把后端接口代理到 api.py（FastAPI），避免跨域；生产可改为同源部署
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      // 前端实际用到的后端接口全部代理到 api.py（FastAPI:8080），避免跨域
      '/answer/stream': { target: 'http://127.0.0.1:8080', changeOrigin: true },
      '/login': 'http://127.0.0.1:8080',
      '/sessions': 'http://127.0.0.1:8080', // 含 /sessions/rename
      '/revoke': 'http://127.0.0.1:8080',
      '/history': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
    },
  },
})