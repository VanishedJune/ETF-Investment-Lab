import { createRouter, createWebHistory } from "vue-router";
import ResearchWorkbenchView from "./views/ResearchWorkbenchView.vue";
import EtfSlotManagerView from "./views/EtfSlotManagerView.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: ResearchWorkbenchView, meta: { title: "指数与 ETF 行情面板" } },
    { path: "/etf-slots", component: EtfSlotManagerView, meta: { title: "ETF 替换" } },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});
