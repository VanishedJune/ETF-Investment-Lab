import { createRouter, createWebHistory } from "vue-router";
import ResearchWorkbenchView from "./views/ResearchWorkbenchView.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: ResearchWorkbenchView, meta: { title: "双市场周线研究台" } },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});
