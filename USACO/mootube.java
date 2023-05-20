import java.util.*;
import java.io.*;
import java.lang.*;

public class mootube
{
   static int n, q;
   static ArrayList[] vids;
   static boolean[] visited;
   static int count;
   static long lim;
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("mootube.in"));
      n = in.nextInt();
      q = in.nextInt();
      vids = new ArrayList[n];
      for(int i = 0; i < n; i++) vids[i] = new ArrayList<pair>();
      for(int i = 0; i < n; i++) vids[i].add(new pair(i, Long.MAX_VALUE));
      for(int i = 0; i < n - 1; i++)
      {
         int a = in.nextInt() - 1, b = in.nextInt() - 1;
         long r = in.nextLong();
         vids[a].add(new pair(b, r));
         vids[b].add(new pair(a, r));
      }
      
      PrintWriter out = new PrintWriter(new File("mootube.out"));
      for(int i = 0; i < q; i++)
      {
         count = 0;
         visited = new boolean[n];
         lim = in.nextLong();
         int v = in.nextInt() - 1;
         ff(v, v, Long.MAX_VALUE);
         // System.out.println(count - 1);
         out.println(count - 1);
      }
      
      in.close();
      out.close();
   }
   
   public static void ff(int start, int current, long min)
   {
      if(visited[current] || min < lim) 
         return;
      visited[current] = true;
      count++;
      for(int i = 0; i < vids[current].size(); i++)
         if(!visited[((pair)(vids[current].get(i))).vid])
            ff(current, ((pair)(vids[current].get(i))).vid, Math.min(((pair)(vids[current].get(i))).rel, min));
   }
   
   static class pair
   {
      public int vid;
      public long rel;
      public pair(int a, long b)
      {
         vid = a;
         rel = b;
      }
   }
}