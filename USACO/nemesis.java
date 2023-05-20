import java.util.*;
import java.io.*;
import java.lang.*;
public class nemesis
{
   static int n, m;
   static ArrayList[] a;
   static int[] ans;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("nemesis.in"));
      n = in.nextInt();
      m = in.nextInt();
      a = new ArrayList[m];
      for(int i = 0; i < m; i++)
         a[i] = new ArrayList<Integer>();
        
      for(int i = 0; i < m; i++)
      {
         int t1 = in.nextInt(), t2 = in.nextInt();
         t1--;
         t2--;
         a[t1].add(t2);
         a[t2].add(t1);
      }
        
      ans = new int[m];
      boolean[] visited = new boolean[m];
      visited[0] = true;
      Queue<Integer> qpos = new LinkedList<>();
      Queue<Integer> qdist = new LinkedList<>();
      qpos.add(0);
      qdist.add(0);
      while(!qpos.isEmpty())
      {
         int cur = qpos.remove();
         int dist = qdist.remove();
         if(ans[cur] != 0) ans[cur] = Math.min(ans[cur], dist);
         else ans[cur] = dist;
         for(int i = 0; i < a[cur].size(); i++)
            if(!visited[(int)a[cur].get(i)])
            {
               visited[(int)a[cur].get(i)] = true;
               qpos.add((int)a[cur].get(i));
               qdist.add(dist + 1);
            }
      }
      
      int maxind = 0;
      int val = 0;
      int freq = 0;
      for(int i = 0; i < ans.length; i++)
        if(ans[i] > ans[maxind]) maxind = i;
      for(int i = 0; i < ans.length; i++)
        if(ans[i] == ans[maxind]) freq++;
      val = ans[maxind];
      maxind++;
      System.out.println(maxind + " " + val + " " + freq);
   }
}